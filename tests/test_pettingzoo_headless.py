from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _reload_mpe_env_module():
    import env.pettingzoo.mpe_env as mpe_env

    return importlib.reload(mpe_env)


def test_mpe_env_module_configures_headless_sdl(monkeypatch):
    monkeypatch.delenv("DIMA_PETTINGZOO_RENDER", raising=False)
    monkeypatch.delenv("SDL_VIDEODRIVER", raising=False)
    monkeypatch.delenv("SDL_AUDIODRIVER", raising=False)

    _reload_mpe_env_module()

    assert os.environ["SDL_VIDEODRIVER"] == "dummy"
    assert os.environ["SDL_AUDIODRIVER"] == "dummy"


def test_headless_pygame_context_initializes_freetype_without_full_pygame(
    monkeypatch,
):
    monkeypatch.delenv("DIMA_PETTINGZOO_RENDER", raising=False)
    mpe_env = _reload_mpe_env_module()
    calls = []

    def full_pygame_init():
        calls.append("pygame.init")
        return (0, 0)

    def freetype_init():
        calls.append("pygame.freetype.init")

    monkeypatch.setattr(mpe_env.pygame, "init", full_pygame_init)
    monkeypatch.setattr(mpe_env.pygame.freetype, "init", freetype_init)

    with mpe_env._skip_full_pygame_init_when_headless():
        assert mpe_env.pygame.init() == (1, 0)

    assert calls == ["pygame.freetype.init"]
    assert mpe_env.pygame.init is full_pygame_init
