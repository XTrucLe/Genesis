import asyncio
import numpy as np
import torch
import torch.distributed as dist
import fsspec
from huggingface_hub import HfApi
from torch.utils.data import DataLoader, IterableDataset

class BinDataset(IterableDataset):
    def __init__(self, repo_id, block_size, split="train", token=None, chunk_size=32 * 1024 * 1024, shuffle_buffer=512):
        self.repo_id = repo_id
        self.block_size = block_size
        self.split = split
        self.token = token
        self.chunk_size = chunk_size
        self.shuffle_buffer = shuffle_buffer

        api = HfApi(token=self.token)
        try:
            all_files = api.list_repo_files(repo_id=self.repo_id, repo_type="dataset")
        except Exception as e:
            raise RuntimeError(f"Không thể kết nối hoặc quét repo {self.repo_id}. Lỗi: {e}")

        self.bin_files = sorted([f for f in all_files if f.endswith('.bin') and self.split in f])
        if not self.bin_files:
            self.bin_files = sorted([f for f in all_files if f.endswith('.bin')])
        if not self.bin_files:
            raise ValueError(f"Không tìm thấy file .bin nào trong repo {self.repo_id} cho split {self.split}")

        self._stride = self.block_size
        self._sample_len = self.block_size + 1

    @staticmethod
    def _read_file(url: str, token, chunk_size: int):
        storage_options = {"token": token} if token else {}
        with fsspec.open(url, "rb", **storage_options) as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                yield chunk

    def _parse_buffer(self, buf: memoryview) -> torch.Tensor | None:
        n_uint16 = len(buf) // 2
        if n_uint16 < self._sample_len:
            return None, buf

        arr = np.frombuffer(buf, dtype=np.uint16)

        n_samples = (n_uint16 - self._sample_len) // self._stride + 1
        used_elems = (n_samples - 1) * self._stride + self._sample_len
        remainder_bytes = bytes(buf[used_elems * 2:])

        from numpy.lib.stride_tricks import as_strided
        item_bytes = arr.itemsize
        windows = as_strided(
            arr,
            shape=(n_samples, self._sample_len),
            strides=(self._stride * item_bytes, item_bytes),
        )

        t = torch.from_numpy(windows.copy()).long()
        return t, remainder_bytes

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if dist.is_available() and dist.is_initialized():
            ddp_rank, ddp_world = dist.get_rank(), dist.get_world_size()
        else:
            ddp_rank, ddp_world = 0, 1

        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1
        total_shards = ddp_world * num_workers
        shard_id = ddp_rank * num_workers + worker_id

        sharded_files = [f for i, f in enumerate(self.bin_files) if i % total_shards == shard_id]
        rng = np.random.default_rng(seed=torch.initial_seed() + shard_id)

        buf: list[tuple[torch.Tensor, torch.Tensor]] = []

        while True:
            rng.shuffle(sharded_files)
            for file in sharded_files:
                url = f"hf://datasets/{self.repo_id}/{file}"
                remainder = b""

                loop = asyncio.new_event_loop()
                try:
                    async def process_file():
                        nonlocal remainder
                        for chunk in self._read_file(url, self.token, self.chunk_size):
                            data = remainder + chunk
                            mv = memoryview(data)
                            batch_t, remainder = self._parse_buffer(mv)
                            if batch_t is None:
                                continue

                            for i in range(batch_t.shape[0]):
                                row = batch_t[i]
                                x, y = row[:-1], row[1:]
                                buf.append((x, y))
                                if len(buf) >= self.shuffle_buffer:
                                    idx = rng.integers(len(buf))
                                    yield buf.pop(int(idx))
                    
                    gen = process_file()
                    try:
                        while True:
                            item = loop.run_until_complete(gen.__anext__())
                            yield item
                    except StopAsyncIteration:
                        pass
                finally:
                    loop.close()

            rng.shuffle(buf)
            yield from buf
            buf.clear()


class DataModule:
    def __init__(self, cfg: dict):
        self.cfg = cfg

    def build_loader(self) -> DataLoader:
        ds = BinDataset(
            repo_id=self.cfg["hf_dataset_repo"],
            block_size=self.cfg["block_size"],
            split=self.cfg.get("data_split", "train"),
            token=self.cfg.get("hf_token", None),
            chunk_size=self.cfg.get("chunk_size", 32 * 1024 * 1024),
            shuffle_buffer=self.cfg.get("shuffle_buffer", 512),
        )

        return DataLoader(
            ds,
            batch_size=self.cfg["batch_size"],
            num_workers=self.cfg["num_workers"],
            pin_memory=True,
            drop_last=True,
            persistent_workers=self.cfg["num_workers"] > 0,
            prefetch_factor=self.cfg["prefetch_factor"] if self.cfg["num_workers"] > 0 else None,
        )
    