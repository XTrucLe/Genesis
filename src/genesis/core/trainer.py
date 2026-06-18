import os
import torch
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

    def _build_model(self) -> Genesis:
        return Genesis(
            vocab_size=self.cfg["vocab_size"],
            block_size=self.cfg["block_size"],
            layers=self.cfg["layers"],
            heads=self.cfg["heads"],
            dim=self.cfg["dim"],
            lora_rank=self.cfg["lora_rank"],
            rope_dim=self.cfg.get("rope_dim", 64),
            dropout=self.cfg["dropout"],
            grad_checkpoint=self.cfg["grad_checkpoint"],
        )

    def _build_optimizer(self, model: Genesis) -> bnb.optim.AdamW8bit:
        decay, no_decay = [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue

            is_no_decay = p.ndim < 2 or "bias" in name or "norm" in name
            (no_decay if is_no_decay else decay).append(p)
        return bnb.optim.AdamW8bit(
            [
                {"params": decay, "weight_decay": self.cfg["weight_decay"]},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=self.cfg["lr"],
            betas=(0.9, 0.95),
        )

    def _setup_device_env(self):
        ddp = int(os.environ.get("RANK", -1)) != -1
        if ddp:
            dist.init_process_group(backend="nccl")
            local_rank = int(os.environ["LOCAL_RANK"])
            device = torch.device(f"cuda:{local_rank}")
            master_process = int(os.environ["RANK"]) == 0
            world_size = int(os.environ["WORLD_SIZE"])
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            master_process = True
            world_size = 1

        if device.type == "cuda":
            torch.cuda.manual_seed(self.cfg["seed"])
            if IS_AMPERE:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

        torch.manual_seed(self.cfg["seed"])
        return device, master_process, world_size, local_rank if ddp else None

    def run(self):
        device, master_process, ddp_world_size, ddp_local_rank = self._setup_device_env()
        ddp = ddp_world_size > 1
        is_cuda = device.type == "cuda"

        model = self._build_model().to(device)
        optimizer = self._build_optimizer(model)

        amp_dtype = torch.float16 if self.cfg["dtype"] == "float16" else torch.bfloat16

        scaler = torch.amp.GradScaler(enabled=(is_cuda and self.cfg["dtype"] == "float16"))
        amp_ctx = torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=is_cuda)

        loader = self.data_manager.build_loader()
        start_step = step = (
            self.checkpoint_manager.load(
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                dataset=loader.dataset,
                ddp_world_size=ddp_world_size,
            )
            + 1
            if self.cfg["resume"]
            else 0
        )
        raw_model = model

        if self.cfg["compile"] and hasattr(torch, "compile"):
            if master_process:
                print("Compiling model via Inductor (max-autotune)...")
            torch._inductor.config.triton.cudagraphs = False
            torch._inductor.config.coordinate_descent_tuning = True
            model = torch.compile(model)

        if ddp:
            model = DDP(model, device_ids=[ddp_local_rank])
            raw_model = model.module

        data_iter = iter(loader)
        model.train()

        t0, t1 = None, None
        if is_cuda:
            t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

        tokens_per_step = self.cfg["batch_size"] * self.cfg["block_size"] * self.cfg["grad_accum"] * ddp_world_size
        accum_loss_tensor = torch.zeros(1, device=device)
        loss_tensor = torch.zeros(1, device=device) if ddp else None

        if master_process:
            print("─" * 56)
            print("🚀 MODEL CONFIG :")
            print(f"   • Architecture: Dim = {self.cfg['dim']} | Layers = {self.cfg['layers']} | Heads = {self.cfg['heads']}")
            print(f"   • Context Len : {self.cfg['block_size']} tokens")
            print(f"   • Total Params: {model.num_params()}")
            print(f"   • Gradient Checkpt: {'Enabled' if self.cfg['grad_checkpoint'] else 'Disabled'}")
            print("\n⚙️ TRAINING CONFIG:")
            print(f"   • Device/GPU  : {torch.cuda.get_device_name(device) if is_cuda else 'CPU'}")
            print(f"   • Precision   : {self.cfg['dtype'].upper()} (Scaler: {'ON' if scaler.is_enabled() else 'OFF'})")
            print(f"   • Total Batch : {tokens_per_step:,} tokens/step (Accum: {self.cfg['grad_accum']})")
            print(f"   • DDP Mode    : {'Active' if ddp else 'Disabled'} ({ddp_world_size} GPU{'' if ddp_world_size == 1 else 's'})")
            steps_info = f"{step:,} → {start_step + self.cfg['short_run_steps']:,} [SHORT RUN]" if self.cfg["short_run"] else f"{step:,} → {self.cfg['total_steps']:,}"
            print(f"   • Total Steps : {steps_info}")
            print("─" * 56 + "\n")

        while step < self.cfg["total_steps"]:
            lr = get_lr(step, self.cfg)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            do_log = step % self.cfg["log_every"] == 0
            do_save = step > 0 and step % self.cfg["save_every"] == 0

            optimizer.zero_grad(set_to_none=True)
            accum_loss_tensor.zero_()

            if is_cuda and do_log and master_process:
                t0.record()

            for _ in range(self.cfg["grad_accum"]):
                if ddp:
                    model.require_backward_grad_sync = _ == self.cfg["grad_accum"] - 1

                try:
                    x, y = next(data_iter)
                except StopIteration:
                    data_iter = iter(loader)
                    x, y = next(data_iter)

                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

                with amp_ctx:
                    _, loss = model(x, y)

                scaled_loss = loss / self.cfg["grad_accum"]
                (scaler.scale(scaled_loss).backward() if scaler.is_enabled() else scaled_loss.backward())
                accum_loss_tensor += scaled_loss.detach()

            if ddp:
                loss_tensor.fill_(accum_loss_tensor)
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
                accum_loss_val = loss_tensor.item()
            else:
                accum_loss_val = accum_loss_tensor.item()

            if scaler.is_enabled():
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.cfg["max_grad_norm"])
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.cfg["max_grad_norm"])
                optimizer.step()

            if do_log and master_process:
                if is_cuda:
                    t1.record()
                    torch.cuda.synchronize()
                    ms = t0.elapsed_time(t1)
                    msg = f"step {step:7d} | loss {accum_loss_val:.5f} | lr {lr:.2e} | {tokens_per_step / ms:.1f}k tok/s | {ms:.0f}ms"
                else:
                    msg = f"step {step:7d} | loss {accum_loss_val:.5f} | lr {lr:.2e}"
                task_queue.put({"type": "log", "data": msg})

            if self.cfg["short_run"] and step >= start_step + self.cfg["short_run_steps"]:
                if master_process:
                    print(f"Short run complete: ran {self.cfg['short_run_steps']} steps starting from step {start_step}.")
                break

            if do_save and master_process:
                self.checkpoint_manager.save(raw_model, optimizer, scaler, step, accum_loss_val, ddp_world_size)

            step += 1

        if ddp:
            dist.barrier()

        if master_process:
            self.checkpoint_manager.save(raw_model, optimizer, scaler, step, accum_loss_val, ddp_world_size)
            task_queue.join()
            print("Training complete")

        if ddp:
            dist.barrier()
            dist.destroy_process_group()
