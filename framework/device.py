"""Shared compute-device resolution for all HuggingFace/torch code paths.

Honors the FRAMEWORK_DEVICE env var (auto|cpu|cuda), which main.py sets from
compute.device in config. In `auto`, picks GPU only if it is both available AND
its compute capability is in this torch build's supported arch list — this
guards against old cards (e.g. sm_50) that pass is_available() but then crash
with cudaErrorNoKernelImageForDevice.
"""
import os


def resolve_device() -> str:
    """Return 'cpu' or 'cuda' for use with torch / .to(device)."""
    import torch

    pref = os.environ.get("FRAMEWORK_DEVICE", "auto").lower()
    if pref == "cpu":
        return "cpu"
    if pref == "cuda":
        return "cuda"
    if not torch.cuda.is_available():
        return "cpu"
    try:
        major, minor = torch.cuda.get_device_capability(0)
        sm = f"sm_{major}{minor}"
        if sm not in torch.cuda.get_arch_list():
            print(
                f"[INFO] GPU capability {sm} not in torch arch list "
                f"{torch.cuda.get_arch_list()} — falling back to CPU."
            )
            return "cpu"
    except Exception:
        pass
    return "cuda"


def pipeline_device_id() -> int:
    """transformers-pipeline device id: -1 = CPU, 0 = GPU."""
    return 0 if resolve_device() == "cuda" else -1
