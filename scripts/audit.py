"""Print host facts and execute a small gradient check on each available device."""
import json
import platform
import shutil
import sys
import subprocess

import psutil
import torch


def probe(device):
    values = torch.tensor([[1.0, 2.0]], device=device, requires_grad=True)
    loss = (values @ values.T).sum()
    loss.backward()
    assert values.grad.tolist() == [[2.0, 4.0]]
    return {"device": device, "loss": loss.item(), "gradient": values.grad.tolist()}


if __name__ == "__main__":
    available = torch.backends.mps.is_available()
    hardware = {}
    if sys.platform == "darwin":
        report = subprocess.check_output(["system_profiler", "SPDisplaysDataType", "-json"])
        gpu = json.loads(report)["SPDisplaysDataType"][0]
        hardware = {"chip": gpu.get("sppci_model"), "gpu_cores": gpu.get("sppci_cores")}
    print(json.dumps({
        **hardware,
        "python": sys.version.split()[0],
        "os": platform.platform(),
        "cpu_cores": psutil.cpu_count(logical=False),
        "ram_bytes": psutil.virtual_memory().total,
        "disk_free_bytes": shutil.disk_usage(".").free,
        "torch": str(torch.__version__),
        "mps_built": torch.backends.mps.is_built(),
        "mps_available": available,
        "probes": [probe("cpu")] + ([probe("mps")] if available else []),
    }, indent=2))
