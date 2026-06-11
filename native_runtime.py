"""Native runtime setup for MuJoCo and JAX/CUDA training processes."""

import os
import sys
from pathlib import Path
from typing import Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple


_READY_ENV = "DIMA_NATIVE_RUNTIME_READY"
_DISABLE_ENV = "DIMA_DISABLE_NATIVE_RUNTIME"
_XLA_WORKAROUNDS = (
    "--xla_gpu_enable_triton_gemm=false",
    "--xla_disable_hlo_passes=gemm-fusion-autotuner",
    "--xla_gpu_autotune_level=0",
)


def _dedupe_existing_dirs(paths: Iterable[Path]) -> Sequence[str]:
    seen = set()
    result = []
    for path in paths:
        try:
            resolved = path.expanduser()
        except RuntimeError:
            continue
        if not resolved.is_dir():
            continue
        path_str = str(resolved)
        if path_str not in seen:
            seen.add(path_str)
            result.append(path_str)
    return result


def _native_library_dirs(
    *,
    prefix: Optional[Path] = None,
    py_version: Optional[str] = None,
    home: Optional[Path] = None,
) -> Sequence[str]:
    prefix = Path(sys.prefix) if prefix is None else Path(prefix)
    py_version = (
        f"python{sys.version_info.major}.{sys.version_info.minor}"
        if py_version is None
        else py_version
    )
    home = Path.home() if home is None else Path(home)
    site_packages = prefix / "lib" / py_version / "site-packages"

    candidates = []
    for env_name in ("MUJOCO_PY_MUJOCO_PATH", "MUJOCO_PATH"):
        mujoco_root = os.environ.get(env_name)
        if mujoco_root:
            candidates.append(Path(mujoco_root) / "bin")
            candidates.append(Path(mujoco_root) / "lib")

    candidates.extend(
        (
            home / ".mujoco" / "mujoco210" / "bin",
            home / ".mujoco" / "mjpro150" / "bin",
        )
    )

    nvidia_root = site_packages / "nvidia"
    if nvidia_root.is_dir():
        candidates.extend(sorted(nvidia_root.glob("cu*/lib")))
        candidates.extend(sorted(nvidia_root.glob("*/lib")))

    candidates.append(prefix / "lib")

    for cuda_root in sorted(Path("/usr/local").glob("cuda*")):
        candidates.append(cuda_root / "lib64")
        candidates.append(cuda_root / "targets" / "x86_64-linux" / "lib")

    return _dedupe_existing_dirs(candidates)


def _native_binary_dirs(
    *,
    prefix: Optional[Path] = None,
    py_version: Optional[str] = None,
) -> Sequence[str]:
    prefix = Path(sys.prefix) if prefix is None else Path(prefix)
    py_version = (
        f"python{sys.version_info.major}.{sys.version_info.minor}"
        if py_version is None
        else py_version
    )
    site_packages = prefix / "lib" / py_version / "site-packages"
    nvidia_root = site_packages / "nvidia"

    candidates = [prefix / "bin"]
    if nvidia_root.is_dir():
        candidates.extend(sorted(nvidia_root.glob("*/bin")))
    candidates.append(site_packages / "triton" / "backends" / "nvidia" / "bin")
    return _dedupe_existing_dirs(candidates)


def _prepend_paths(current: str, candidates: Sequence[str]) -> str:
    current_entries = [entry for entry in current.split(os.pathsep) if entry]
    current_set = set(current_entries)
    prepended = [entry for entry in candidates if entry not in current_set]
    return os.pathsep.join(prepended + current_entries)


def _append_missing_xla_flags(current: str) -> str:
    entries = [entry for entry in current.split() if entry]
    entries_set = set(entries)
    for flag in _XLA_WORKAROUNDS:
        if flag not in entries_set:
            entries.append(flag)
    return " ".join(entries)


def _is_train_entrypoint(argv: Sequence[str]) -> bool:
    if not argv:
        return False
    return Path(argv[0]).name == "train.py"


def _build_native_runtime_env(
    *,
    env: Mapping[str, str],
    argv: Sequence[str],
    prefix: Path,
    py_version: str,
    home: Path,
) -> Tuple[Mapping[str, str], bool]:
    if env.get(_DISABLE_ENV) == "1" or not _is_train_entrypoint(argv):
        return env, False

    updated: MutableMapping[str, str] = dict(env)
    old_ld_library_path = updated.get("LD_LIBRARY_PATH", "")
    updated["LD_LIBRARY_PATH"] = _prepend_paths(
        old_ld_library_path,
        _native_library_dirs(prefix=prefix, py_version=py_version, home=home),
    )
    updated["PATH"] = _prepend_paths(
        updated.get("PATH", ""),
        _native_binary_dirs(prefix=prefix, py_version=py_version),
    )
    updated["XLA_FLAGS"] = _append_missing_xla_flags(updated.get("XLA_FLAGS", ""))
    updated.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    updated.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    updated.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

    needs_reexec = (
        updated["LD_LIBRARY_PATH"] != old_ld_library_path
        and updated.get(_READY_ENV) != "1"
    )
    if needs_reexec:
        updated[_READY_ENV] = "1"
    return updated, needs_reexec


def prepare_train_process() -> None:
    py_version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    updated_env, should_reexec = _build_native_runtime_env(
        env=os.environ,
        argv=sys.argv,
        prefix=Path(sys.prefix),
        py_version=py_version,
        home=Path.home(),
    )
    if updated_env is not os.environ:
        os.environ.update(updated_env)
    if should_reexec:
        os.execvpe(sys.executable, [sys.executable] + sys.argv, updated_env)
