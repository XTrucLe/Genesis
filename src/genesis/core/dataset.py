import asyncio
import fsspec
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
import numpy as np
from numpy.lib.stride_tricks import as_strided
from huggingface_hub import HfApi

class BinDataset(IterableDataset):
    def __init__(self, repo_id, block_size, split="train", token=None, chunk_size=32, shuffle_buffer=512):
        self.repo_id = repo_id
        self.block_size = block_size
        self.split = split
        self.token = token

        self.chunk_size = chunk_size * 1024 * 1024 # MB
        self.shuffle_buffer = shuffle_buffer
        self.shards = []

        self.api = HfApi(token=self.token)
        try:
            all_files = self.api.list_repo_files(repo_id=self.repo_id, repo_type="dataset")
        except Exception as e:
            raise RuntimeError(f"Can't reach the repo {self.repo_id}. Error: {e}")

        self.bin_files = sorted([f for f in all_files if f.endswith('.bin') and self.split in f])
        if not self.bin_files:
            self.bin_files = sorted([f for f in all_files if f.endswith('.bin')])
        if not self.bin_files:
            raise ValueError(f"Can't find any .bin files in repo {self.repo_id} for split {self.split}")

        self._stride = self.block_size
        self._sample_len = self.block_size + 1

        self.samples_offset = 0

        self._build_metadata_index()

    def set_resume_state(self, samples_offset: int):
        self.samples_offset = samples_offset

    
    def load_state_dict(self, state):
        self.current_shard_idx = state["current_shard_idx"]
        self.samples_offset = state["samples_offset"]

    def _get_file_size(self, file_path: str) -> int:
        try:
            file_info = self.api.get_paths_info(
                repo_id=self.repo_id,
                repo_type="dataset",
                paths=[file_path]
            )
            return file_info[0].size
        except Exception as e:
            raise RuntimeError(f"Can't get the size of file {file_path} from Hugging Face. Error: {e}")

    def _build_metadata_index(self):
        self.shards = []
        samples_per_shard = max(1, (self.chunk_size // 2 - self._sample_len) // self._stride + 1)

        print("Building metadata index from Hugging Face...")
        for file_path in self.bin_files:
            file_size = self._get_file_size(file_path)
            
            n_tokens = file_size // 2
            n_samples = (n_tokens - self._sample_len) // self._stride + 1
            
            if n_samples <= 0:
                continue
                
            for i in range(0, n_samples, samples_per_shard):
                shard_samples = min(samples_per_shard, n_samples - i)
                start_byte = (i * self._stride) * 2
                
                shard_bytes = ((shard_samples - 1) * self._stride + self._sample_len) * 2
                
                self.shards.append({
                    "file_path": file_path,
                    "start_byte": start_byte,
                    "size": shard_bytes,
                    "num_samples": shard_samples
                })
            
        print(f"Building completed! Total samples: {len(self.shards)}")

    @staticmethod
    def _read_shard(url: str, token, start_byte: int, read_size: int, buffer_size: int = 4 * 1024 * 1024):
        storage_options = {"token": token} if token else {}
        with fsspec.open(url, "rb", **storage_options) as f:
            f.seek(start_byte)
            bytes_left = read_size
            while bytes_left > 0:
                to_read = min(buffer_size, bytes_left)
                chunk = f.read(to_read)
                if not chunk:
                    break
                bytes_left -= len(chunk)
                yield chunk

    def _parse_buffer(self, buf: memoryview) -> tuple[torch.Tensor | None, bytes]:
        n_uint16 = len(buf) // 2
        if n_uint16 < self._sample_len:
            return None, bytes(buf)

        arr = np.frombuffer(buf, dtype=np.uint16)

        n_samples = (n_uint16 - self._sample_len) // self._stride + 1
        used_elems = (n_samples - 1) * self._stride + self._sample_len
        remainder_bytes = bytes(buf[used_elems * 2:])

        item_bytes = arr.itemsize
        windows = as_strided(
            arr,
            shape=(n_samples, self._sample_len),
            strides=(self._stride * item_bytes, item_bytes),
        )

        t = torch.from_numpy(windows.copy()).long()
        return t, remainder_bytes

    def __iter__(self):
        worker_info = get_worker_info()
        if dist.is_available() and dist.is_initialized():
            ddp_rank, ddp_world = dist.get_rank(), dist.get_world_size()
        else:
            ddp_rank, ddp_world = 0, 1

        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1

        total_workers = ddp_world * num_workers
        global_worker_id = ddp_rank * num_workers + worker_id

        worker_shards = [shard for i, shard in enumerate(self.shards) if i % total_workers == global_worker_id]
        rng = np.random.default_rng(seed=torch.initial_seed() + global_worker_id)

        buf: list[tuple[torch.Tensor, torch.Tensor]] = []

        global_samples_to_skip = self.samples_offset

        while True:
            for shard_idx, shard in enumerate(worker_shards):
                if global_samples_to_skip >= shard["num_samples"]:
                    global_samples_to_skip -= shard["num_samples"]
                    continue

                actual_start_byte = shard["start_byte"]
                actual_size = shard["size"]

                if global_samples_to_skip > 0:
                    byte_offset = (global_samples_to_skip * self._stride) * 2
                    actual_start_byte += byte_offset
                    actual_size -= byte_offset
                    global_samples_to_skip = 0

                url = f"hf://datasets/{self.repo_id}/{shard['file_path']}"
                remainder = b""

                for chunk in self._read_shard(url, self.token, actual_start_byte, actual_size):
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
        
            if buf:
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
            chunk_size=self.cfg.get("chunk_size", 32),
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
    