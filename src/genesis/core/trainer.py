import os
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import bitsandbytes as bnb
from genesis.core.model.Genesis import Genesis
from genesis.utils.common import IS_AMPERE, get_lr, task_queue


class Trainer:
    def __init__(self, cfg: dict, data_manager, checkpoint_manager):
        self.cfg = cfg
        self.data_manager = data_manager
        self.checkpoint_manager = checkpoint_manager

    def _build_model(self, device):
        return Genesis(
            vocab_size=self.cfg["vocab_size"],
            block_size=self.cfg["block_size"],
            layers=self.cfg["layers"],
            heads=self.cfg["heads"],
            dim=self.cfg["dim"],
            kv_heads=self.cfg["kv_heads"],
            dropout=self.cfg["dropout"],
            grad_checkpoint=self.cfg["grad_checkpoint"],
        ).to(device)

    def _build_optimizer(self, model):
        decay, no_decay = [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            (decay if p.ndim >= 2 and "embedding" not in name else no_decay).append(p)
        return bnb.optim.AdamW8bit(
            [
                {"params": decay, "weight_decay": self.cfg["weight_decay"]},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=self.cfg["lr"],
            betas=(0.9, 0.95),
        )

    def run(self):
        ddp = int(os.environ.get("RANK", -1)) != -1
        if ddp:
            dist.init_process_group(backend="nccl")
            ddp_rank = int(os.environ["RANK"])
            ddp_local_rank = int(os.environ["LOCAL_RANK"])
            ddp_world_size = int(os.environ["WORLD_SIZE"])
            device = torch.device(f"cuda:{ddp_local_rank}")
            torch.cuda.set_device(device)
            master_process = ddp_rank == 0
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            master_process = True
            ddp_world_size = 1
            ddp_local_rank = 0

        is_cuda = device.type == "cuda"

        if IS_AMPERE:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

        torch.manual_seed(self.cfg["seed"])
        if is_cuda:
            torch.cuda.manual_seed(self.cfg["seed"])

        model = self._build_model(device)
        optimizer = self._build_optimizer(model)

        if master_process:
            print(f"GPU   : {torch.cuda.get_device_name(device) if is_cuda else 'CPU'}")
            print(f"Params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
            print(f"dtype : {self.cfg['dtype']}")
            print(
                f"DDP   : {'Active' if ddp else 'Disabled'} (Number of GPUs: {ddp_world_size})"
            )

        use_scaler = is_cuda and self.cfg["dtype"] == "float16"
        scaler = torch.amp.GradScaler(enabled=use_scaler)
        amp_dtype = torch.float16 if self.cfg["dtype"] == "float16" else torch.bfloat16
        amp_ctx = torch.amp.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=is_cuda
        )

        ckpt_mgr = self.checkpoint_manager
        loader = self.data_manager.build_loader()
        step = (
            ckpt_mgr.load(
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                dataset=loader.dataset,
                ddp_world_size=ddp_world_size,
            )
            if self.cfg["resume"]
            else 0
        )

        if ddp:
            model = DDP(model, device_ids=[ddp_local_rank])

        raw_model = model.module if ddp else model

        if self.cfg["compile"] and hasattr(torch, "compile"):
            print("Compiling...")
            model = torch.compile(model, mode="reduce-overhead")

        data_iter = iter(loader)
        model.train()

        t0 = t1 = None
        if is_cuda:
            t0 = torch.cuda.Event(enable_timing=True)
            t1 = torch.cuda.Event(enable_timing=True)

        tokens_per_step = (
            self.cfg["batch_size"]
            * self.cfg["block_size"]
            * self.cfg["grad_accum"]
            * ddp_world_size
        )

        if master_process:
            print(f"\nStart training | step={step} → {self.cfg['total_steps']}")
            print(f"Effective batch: {tokens_per_step:,} tokens/step\n")

        while step < self.cfg["total_steps"]:
            lr = get_lr(step, self.cfg)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            do_log = step % self.cfg["log_every"] == 0
            do_save = step > 0 and step % self.cfg["save_every"] == 0

            optimizer.zero_grad(set_to_none=True)
            accum_loss = 0.0

            if is_cuda and do_log and master_process:
                t0.record()

            for _ in range(self.cfg["grad_accum"]):
                if ddp:
                    model.require_backward_grad_sync = _ == self.cfg["grad_accum"] - 1

                x, y = next(data_iter)
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                with amp_ctx:
                    loss = model(x, y)
                    scaled_loss = loss / self.cfg["grad_accum"]

                scaler.scale(scaled_loss).backward()
                accum_loss += loss.detach().item()

            if ddp:
                loss_tensor = torch.tensor(accum_loss, device=device)
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
                accum_loss = loss_tensor.item()

            accum_loss /= self.cfg["grad_accum"]

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), self.cfg["max_grad_norm"]
            )
            scaler.step(optimizer)
            scaler.update()

            if do_log and master_process:
                if is_cuda:
                    t1.record()
                    torch.cuda.synchronize()
                    ms = t0.elapsed_time(t1)
                    msg = (
                        f"step {step:7d} | loss {accum_loss:.5f} | lr {lr:.2e} "
                        f"| {tokens_per_step / ms:.1f}k tok/s | {ms:.0f}ms"
                    )
                else:
                    msg = f"step {step:7d} | loss {accum_loss:.5f} | lr {lr:.2e}"
                task_queue.put({"type": "log", "data": msg})

            if do_save and master_process:
                self.checkpoint_manager.save(
                    raw_model, optimizer, scaler, step, accum_loss, ddp_world_size
                )

            step += 1

        if master_process:
            self.checkpoint_manager.save(
                raw_model, optimizer, scaler, step, 0.0, ddp_world_size
            )
            task_queue.join()
            print("Training complete")

        if ddp:
            dist.destroy_process_group()
