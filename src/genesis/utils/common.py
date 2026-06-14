import os, math, re, json, queue, threading
import torch

PATTERN = re.compile(
    r"step\s+(?P<step>\d+)\s+\| "
    r"loss (?P<loss>[0-9.]+)\s+\| "
    r"lr (?P<lr>[0-9.eE+-]+)"
    r"(?: \| (?P<tok_s>[0-9.]+)k tok/s \| (?P<ms>[0-9.]+)ms)?"
)

task_queue = queue.Queue()


def _gpu_info():
    if not torch.cuda.is_available():
        return False, False, False

    major, minor = torch.cuda.get_device_capability(0)

    is_ampere = major >= 8
    is_turing = major == 7
    has_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    return is_ampere, is_turing, has_bf16


IS_AMPERE, IS_TURING, HAS_BF16 = _gpu_info()


def get_lr(step, cfg):
    if step < cfg["warmup_steps"]:
        return cfg["lr"] * step / cfg["warmup_steps"]

    progress = (step - cfg["warmup_steps"]) / max(
        1, cfg["total_steps"] - cfg["warmup_steps"]
    )
    return cfg["min_lr"] + 0.5 * (cfg["lr"] - cfg["min_lr"]) * (
        1 + math.cos(math.pi * progress)
    )


def get_raw_model(model):
    while hasattr(model, "module"):
        model = model.module
    while hasattr(model, "_orig_mod"):
        model = model._orig_mod
    return model


def _to_cpu(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_cpu(v) for v in obj)
    return obj


def write_log(msg, log_file="src/log/logs.jsonl"):
    if not (m := PATTERN.match(msg)):
        return
    d = m.groupdict()

    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "step": int(d["step"]),
                    "loss": float(d["loss"]),
                    "lr": float(d["lr"]),
                    "tok/s": float(d["tok_s"]) * 1000 if d["tok_s"] else None,
                    "ms": float(d["ms"]) if d["ms"] else None,
                }
            )
            + "\n"
        )


def _background_worker():
    while True:
        task = task_queue.get()
        if task is None:
            break
        try:
            if task["type"] == "log":
                write_log(task["data"], "src/log/logs.jsonl")
                print(task["data"])
        except Exception as e:
            print(f"[worker] error: {e}")
        finally:
            task_queue.task_done()


threading.Thread(target=_background_worker, daemon=True).start()
