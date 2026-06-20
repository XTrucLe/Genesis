import os
import json
import torch
from safetensors.torch import save_file, load_file
from huggingface_hub.utils import logging, disable_progress_bars
from genesis.utils.common import get_raw_model, _to_cpu

logging.set_verbosity_error()
disable_progress_bars()


class CheckpointModule:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.repo_id = self.cfg["hf_repo_id"]
        self.token = self.cfg.get("hf_token", os.environ.get("HF_TOKEN"))

        self.local_temp_dir = os.path.join(cfg.get("checkpoint_dir", "checkpoints"), "hf_staging")
        os.makedirs(self.local_temp_dir, exist_ok=True)

        from huggingface_hub import HfApi

        self.api = HfApi()
        try:
            self.api.create_repo(repo_id=self.repo_id, token=self.token, exist_ok=True, private=True)
        except Exception as e:
            print(f"[hf_ckpt] Warning when creating repo: {e}")

    def save(self, model, optimizer, scaler, step, loss, ddp_world_size=1):
        raw_model = get_raw_model(model)

        raw_state = raw_model.state_dict()
        clean_state = {k.removeprefix("_orig_mod.").removeprefix("module."): v for k, v in raw_state.items()}
        cpu_clean_state = _to_cpu(clean_state)
        cpu_clean_state = {k: v.contiguous() for k, v in cpu_clean_state.items()}

        config_data = {
            "architectures": [raw_model.__class__.__name__],
            "vocab_size": self.cfg["vocab_size"],
            "hidden_size": self.cfg["dim"],
            "num_hidden_layers": self.cfg["layers"],
            "num_attention_heads": self.cfg["heads"],
            "max_position_embeddings": self.cfg["block_size"],
            "torch_dtype": self.cfg["dtype"],
        }
        config_path = os.path.join(self.local_temp_dir, "config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2)

        safetensors_path = os.path.join(self.local_temp_dir, "model.safetensors")
        save_file(
            cpu_clean_state,
            safetensors_path,
            metadata={"step": str(step), "loss": f"{loss:.4f}"},
        )

        current_global_batch_size = self.cfg["batch_size"] * self.cfg["grad_accum"] * ddp_world_size
        samples_trained = step * current_global_batch_size

        training_state_path = os.path.join(self.local_temp_dir, "training_state.pt")
        torch.save(
            {
                "step": step,
                "batch_size": self.cfg["batch_size"],
                "grad_accum": self.cfg["grad_accum"],
                "samples_trained": samples_trained,
                "loss": float(loss),
                "optimizer": _to_cpu(optimizer.state_dict()),
                "scaler": _to_cpu(scaler.state_dict()) if scaler.is_enabled() else None,
            },
            training_state_path,
        )

        try:
            self.api.upload_folder(
                folder_path=self.local_temp_dir,
                repo_id=self.repo_id,
                token=self.token,
                commit_message=f"Automated backup - Step {step}",
                run_as_future=True,
            )
            print("[HF Backup] Upload task spawned successfully in background.")
        except Exception as e:
            print(f"❌ [HF Backup] Failed to trigger upload: {e}")

    def load(self, model, optimizer, scaler, dataset=None, ddp_world_size=1) -> int:
        from huggingface_hub import hf_hub_download

        print(f"[hf_ckpt] Checking for the latest checkpoint on Hugging Face Hub ({self.repo_id})...")

        try:
            safetensors_path = hf_hub_download(repo_id=self.repo_id, filename="model.safetensors", token=self.token)
            training_state_path = hf_hub_download(repo_id=self.repo_id, filename="training_state.pt", token=self.token)
        except Exception as e:
            print(f"[hf_ckpt] No valid checkpoint found on the Hub. Starting training from scratch (Step 0). Details: {e}")
            return 0

        state = load_file(safetensors_path, device="cpu")

        raw_model = get_raw_model(model)
        raw_model.load_state_dict(state)

        ckpt = torch.load(training_state_path, map_location="cpu", weights_only=True)
        optimizer.load_state_dict(ckpt["optimizer"])

        if ckpt.get("scaler") and len(ckpt["scaler"]) > 0:
            if hasattr(scaler, "is_enabled") and scaler.is_enabled():
                scaler.load_state_dict(ckpt["scaler"])

        samples_trained = ckpt.get(
            "samples_trained",
            ckpt["step"] * self.cfg["batch_size"] * self.cfg["grad_accum"] * ddp_world_size,
        )

        if dataset and hasattr(dataset, "set_resume_state"):
            dataset.set_resume_state(samples_trained)

        rescaled_step = samples_trained // (self.cfg["batch_size"] * self.cfg["grad_accum"] * ddp_world_size)

        print(f"[hf_ckpt] Successfully loaded checkpoint from Hugging Face Hub at step {rescaled_step}")
        return rescaled_step
