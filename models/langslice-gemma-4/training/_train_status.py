"""One-shot training-status dump.

Reads `out/sft/logs/latest_run.txt` (written by `_run_sft_msvc.cmd`'s caller)
and prints:
- python child process: alive?, RSS, CPU time
- GPU utilization, VRAM, temp, power
- system: free RAM, total RAM
- err-log: kwargs-warning count + last N non-kwargs lines
- output-dir: any checkpoint subdirs

Run from the repo root or set ``LANGSLICE_REPO_ROOT`` explicitly.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(os.environ.get("LANGSLICE_REPO_ROOT", Path.cwd())).resolve()
LOGS = REPO / "out" / "sft" / "logs"
LATEST = LOGS / "latest_run.txt"


def _read_latest_run() -> dict[str, str]:
    if not LATEST.exists():
        return {}
    out: dict[str, str] = {}
    for line in LATEST.read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def _summarize_log(path: Path, *, tail_n: int = 8) -> None:
    if not path.exists():
        print(f"[err] log not found: {path}")
        return
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    kwargs = sum(1 for ln in lines if "Kwargs passed to" in ln)
    non_kwargs = [ln for ln in lines if "Kwargs passed to" not in ln]
    print(
        f"[err] {path.name}: {len(lines)} lines, {kwargs} kwargs warnings, "
        f"{len(non_kwargs)} other"
    )
    print(f"[err] last {tail_n} non-kwargs:")
    for ln in non_kwargs[-tail_n:]:
        print(f"  {ln}")


def _gpu_snapshot() -> None:
    try:
        import pynvml  # type: ignore
    except ImportError:
        try:
            import nvidia_ml_py as pynvml  # type: ignore  # nvitop bundles this
        except ImportError:
            print("[gpu] no NVML bindings; falling back to nvidia-smi")
            query = "utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw"
            os.system(f"nvidia-smi --query-gpu={query} --format=csv,noheader")
            return
    pynvml.nvmlInit()
    n = pynvml.nvmlDeviceGetCount()
    for i in range(n):
        h = pynvml.nvmlDeviceGetHandleByIndex(i)
        name = pynvml.nvmlDeviceGetName(h)
        if isinstance(name, bytes):
            name = name.decode()
        util = pynvml.nvmlDeviceGetUtilizationRates(h)
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        try:
            temp = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
        except pynvml.NVMLError:
            temp = -1
        try:
            power_w = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
        except pynvml.NVMLError:
            power_w = -1
        used_gb = mem.used / 1e9
        total_gb = mem.total / 1e9
        free_gb = mem.free / 1e9
        print(
            f"[gpu] {i} {name}: util {util.gpu:>3}% mem-util {util.memory:>3}% | "
            f"VRAM {used_gb:.2f}/{total_gb:.2f} GB (free {free_gb:.2f}) | "
            f"{temp}°C {power_w:.0f}W"
        )
    pynvml.nvmlShutdown()


def _process_snapshot(pid: int | None) -> None:
    import psutil

    if pid is None:
        print("[proc] no pid in latest_run.txt")
        return
    try:
        p = psutil.Process(pid)
    except psutil.NoSuchProcess:
        # The cmd-shell wrapper exits early; the python child is what matters.
        print(f"[proc] cmd shell pid={pid} no longer alive — looking for orphaned python child")
        return
    children = [c for c in p.children(recursive=True) if c.name().lower() == "python.exe"]
    targets = [p, *children]
    for proc in targets:
        if proc.name().lower() not in ("python.exe", "cmd.exe"):
            continue
        try:
            with proc.oneshot():
                rss_gb = proc.memory_info().rss / 1e9
                cpu_t = proc.cpu_times()
                cpu_total = cpu_t.user + cpu_t.system
                pct = proc.cpu_percent(interval=0.1)
                threads = proc.num_threads()
                status = proc.status()
            print(
                f"[proc] {proc.name()} pid={proc.pid} status={status} "
                f"rss={rss_gb:.2f}GB cpu={cpu_total:.0f}s ({pct:.0f}% inst) "
                f"threads={threads}"
            )
        except psutil.NoSuchProcess:
            print(f"[proc] pid={proc.pid} died mid-snapshot")


def _system_snapshot() -> None:
    import psutil
    vm = psutil.virtual_memory()
    print(
        f"[sys] RAM {vm.used / 1e9:.1f}/{vm.total / 1e9:.1f} GB used "
        f"({vm.percent:.0f}%), free {vm.available / 1e9:.1f} GB"
    )


def _output_dir_snapshot(stdout_path: str | None) -> None:
    if not stdout_path:
        return
    # Output dir is sibling of logs/, named by run_id
    log_path = Path(stdout_path)
    run_id = log_path.stem.split(".")[0]
    out_dir = REPO / "out" / "sft" / run_id
    if not out_dir.exists():
        print(f"[out] {out_dir} (not created yet)")
        return
    entries = sorted(out_dir.iterdir())
    print(f"[out] {out_dir} ({len(entries)} entries)")
    for e in entries[:10]:
        size = e.stat().st_size if e.is_file() else "<dir>"
        print(f"  {e.name}  {size}")


def main() -> None:
    info = _read_latest_run()
    print(f"=== run {info.get('run_id', '?')} | pid {info.get('pid', '?')} ===")
    _process_snapshot(int(info["pid"]) if info.get("pid", "").isdigit() else None)
    _gpu_snapshot()
    _system_snapshot()
    _output_dir_snapshot(info.get("stderr"))
    err_log = info.get("stderr")
    if err_log:
        _summarize_log(Path(err_log))


if __name__ == "__main__":
    main()
