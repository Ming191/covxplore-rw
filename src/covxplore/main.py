#!/usr/bin/env python
"""Backward-compatible console-script entry points."""

from __future__ import annotations

from covxplore.cli import run_ablation, run_generation, run_parallel

__all__ = ["run_generation", "run_ablation", "run_parallel"]
