#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Checkpoint backward compatibility test.

Verifies that checkpoints saved by one git commit can be loaded by another
and produce bit-identical losses and grad norms. Runs three training jobs:

  1. Reference run at <load_commit>: train full --steps from scratch.
  2. Save run at <save_commit>: train --resume-step steps, save checkpoint.
  3. Resume run at <load_commit>: resume from save checkpoint to --steps.

The save run uses --save-config and the other two --load-config, both
defaulting to --config, so the test can also save under one config and resume
under another at the same commit, e.g. under another FSDP backend.

All generated run configs use deterministic execution and seed 42. Pass means
the losses and grad norms are identical.

Examples:
  python scripts/checkpoint_compat_test.py HEAD~1 HEAD
  python scripts/checkpoint_compat_test.py HEAD~1 HEAD --steps=100 --resume-step=50
  python scripts/checkpoint_compat_test.py HEAD~1 HEAD --assert-equal
  python scripts/checkpoint_compat_test.py HEAD~1 . --assert-equal
  python scripts/checkpoint_compat_test.py . . --assert-equal \
      --save-config=llama3_debugmodel_no_cuda_graphs \
      --load-config=llama3_debugmodel_flex_shard
"""

import argparse
import os
import subprocess
import sys
import tempfile

if __package__:
    from scripts._checkpoint_test_config import configure_training_run
else:
    from _checkpoint_test_config import (  # pyrefly: ignore [missing-import]
        configure_training_run,
    )

# TensorBoard scalar tags of the compared metrics.
TB_TAGS = {"loss": "loss_metrics/global_avg_loss", "grad_norm": "grad_norm"}


def log(msg: str = "") -> None:
    print(f"[CKPT_COMPAT] {msg}" if msg else "[CKPT_COMPAT]")


def git_run(*args: str) -> str:
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def check_git_clean() -> None:
    lines = git_run("status", "--porcelain").split("\n")
    modified = [l for l in lines if l and not l.startswith("??")]
    if modified:
        log("Error: uncommitted changes to tracked files:")
        for l in modified:
            log(f"  {l}")
        sys.exit(1)


def get_current_ref() -> str:
    ref = git_run("rev-parse", "--abbrev-ref", "HEAD")
    return ref if ref != "HEAD" else git_run("rev-parse", "HEAD")


def checkout(commit: str, label: str) -> None:
    if commit != ".":
        log(f"Checking out {label}: {commit}")
        subprocess.run(["git", "checkout", commit], check=True)


def run_cmd(
    cmd: str,
    logfile: str,
    ngpus: int,
    env_overrides: dict[str, str] | None = None,
) -> None:
    """Run training command with real-time output and log capture."""
    log(f"Executing: {cmd}")
    env = {**os.environ, "NGPU": str(ngpus), "PYTHONUNBUFFERED": "1"}
    if env_overrides:
        env.update(env_overrides)
    with open(logfile, "w") as f:
        proc = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
            bufsize=1,
        )
        for line in proc.stdout:  # pyrefly: ignore [not-iterable]
            print(line, end="")
            f.write(line)
        proc.wait()
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, cmd)


def build_cmd(
    module: str,
    config: str,
    dump_folder: str,
) -> str:
    return (
        f"MODULE='{module}' CONFIG='{config}' ./run_train.sh"
        f" --output-dir={dump_folder}"
    )


def extract_tb_metrics(tb_base: str) -> dict[str, dict[int, float]]:
    """Extract full-precision metrics from all TB subdirs under tb_base.

    Unlike loss_compare.extract_losses_from_tensorboard (which expects a single
    subdirectory), this merges events across multiple subdirs to handle the
    resume case where save and resume runs write to the same TB folder.
    """
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    metrics: dict[str, dict[int, float]] = {name: {} for name in TB_TAGS}
    for subdir in sorted(os.listdir(tb_base)):
        path = os.path.join(tb_base, subdir)
        if not os.path.isdir(path):
            continue
        acc = EventAccumulator(path)
        acc.Reload()
        tags = acc.Tags().get("scalars", [])
        for name, tag in TB_TAGS.items():
            if tag in tags:  # pyrefly: ignore [not-iterable]
                for s in acc.Scalars(tag):
                    metrics[name][s.step] = s.value
    log(
        f"Extracted {', '.join(f'{len(v)} steps of {k}' for k, v in metrics.items())}"
        f" from {tb_base}"
    )
    return metrics


def compare_metrics(
    ref: dict[str, dict[int, float]],
    resume: dict[str, dict[int, float]],
    assert_equal: bool,
) -> bool:
    all_ok = True
    for name in TB_TAGS:
        ref_steps = sorted(ref[name])
        if not ref_steps or ref_steps != sorted(resume[name]):
            log(
                f"{name} step mismatch: ref={len(ref[name])}, "
                f"resume={len(resume[name])}"
            )
            all_ok = False
            continue

        log()
        log(f"{name}:")
        log(f"{'Step':<8} {'Reference':<22} {'Resume':<22} {'Match'}")
        log("-" * 60)
        for step in ref_steps:
            ok = ref[name][step] == resume[name][step]
            all_ok &= ok
            log(
                f"{step:<8} {ref[name][step]!r:<22} {resume[name][step]!r:<22} "
                f"{'OK' if ok else 'MISMATCH'}"
            )

    log()
    log(
        "PASS: All losses and grad norms are bit-identical."
        if all_ok
        else "FAIL: Loss or grad norm mismatch detected!"
    )

    if assert_equal and not all_ok:
        sys.exit(1)
    return all_ok


def main() -> None:
    p = argparse.ArgumentParser(
        description="Test checkpoint backward compatibility between two git commits.",
    )
    p.add_argument("save_commit", help="Commit that saves the checkpoint (old code)")
    p.add_argument("load_commit", help="Commit that loads the checkpoint (new code)")
    p.add_argument("--steps", type=int, default=100, help="Total training steps")
    p.add_argument("--resume-step", type=int, default=50, help="Checkpoint/resume step")
    p.add_argument("--module", default="torchtitan_recipes.tests.models.llama3")
    p.add_argument("--config", default="llama3_debugmodel")
    p.add_argument(
        "--save-config", default="", help="Config of the save run (default: --config)"
    )
    p.add_argument(
        "--load-config",
        default="",
        help="Config of the reference and resume runs (default: --config)",
    )
    p.add_argument("--ngpus", type=int, default=8)
    p.add_argument("--output-folder", default="")
    p.add_argument(
        "--assert-equal", action="store_true", help="Exit non-zero on mismatch"
    )
    args = p.parse_args()

    if args.resume_step >= args.steps:
        p.error(f"--resume-step ({args.resume_step}) must be < --steps ({args.steps})")
    if not args.output_folder:
        args.output_folder = tempfile.mkdtemp(prefix="ckpt_compat_")
    save_config = args.save_config or args.config
    load_config = args.load_config or args.config

    log("Checkpoint Backward Compatibility Test")
    log(
        f"  save={args.save_commit}  load={args.load_commit}  "
        f"steps={args.steps}  resume_step={args.resume_step}  ngpus={args.ngpus}"
    )
    log(f"  save_config={save_config}  load_config={load_config}")
    log(f"  output: {args.output_folder}")

    # Resolve SHAs before any checkout
    save_sha = (
        git_run("rev-parse", args.save_commit) if args.save_commit != "." else "."
    )
    load_sha = (
        git_run("rev-parse", args.load_commit) if args.load_commit != "." else "."
    )

    ref_dump = os.path.join(args.output_folder, "ref_outputs")
    resume_dump = os.path.join(args.output_folder, "resume_outputs")
    os.makedirs(ref_dump, exist_ok=True)
    os.makedirs(resume_dump, exist_ok=True)

    needs_checkout = save_sha != "." or load_sha != "."
    original = get_current_ref() if needs_checkout else None
    if needs_checkout:
        check_git_clean()

    try:
        # Step 1: Reference run at load_commit (full training from scratch)
        log()
        log("=" * 60)
        log("STEP 1: Reference run")
        log("=" * 60)
        checkout(load_sha, "load_commit")
        ref_env = dict(os.environ)
        ref_module, ref_config = configure_training_run(
            ref_env,
            module=args.module,
            config=load_config,
            steps=args.steps,
            tb_folder="tb",
        )
        cmd = build_cmd(ref_module, ref_config, ref_dump)
        run_cmd(
            cmd,
            os.path.join(args.output_folder, "reference.log"),
            args.ngpus,
            ref_env,
        )

        # Step 2: Save run at save_commit (partial training + checkpoint)
        log()
        log("=" * 60)
        log(f"STEP 2: Save run ({args.resume_step} steps)")
        log("=" * 60)
        checkout(save_sha, "save_commit")
        save_env = dict(os.environ)
        save_module, save_run_config = configure_training_run(
            save_env,
            module=args.module,
            config=save_config,
            steps=args.resume_step,
            tb_folder="tb",
            checkpoint_mode="resume",
            checkpoint_interval=args.resume_step,
            total_steps=args.steps,
        )
        cmd = build_cmd(
            save_module,
            save_run_config,
            resume_dump,
        )
        run_cmd(
            cmd,
            os.path.join(args.output_folder, "save.log"),
            args.ngpus,
            save_env,
        )

        # Step 3: Resume run at load_commit (load checkpoint, train to end)
        log()
        log("=" * 60)
        log(f"STEP 3: Resume run (to step {args.steps})")
        log("=" * 60)
        checkout(load_sha, "load_commit")
        resume_env = dict(os.environ)
        resume_module, resume_config = configure_training_run(
            resume_env,
            module=args.module,
            config=load_config,
            steps=args.steps,
            tb_folder="tb",
            checkpoint_mode="resume",
            checkpoint_interval=args.resume_step,
        )
        cmd = build_cmd(
            resume_module,
            resume_config,
            resume_dump,
        )
        run_cmd(
            cmd,
            os.path.join(args.output_folder, "resume.log"),
            args.ngpus,
            resume_env,
        )

        # Step 4: Compare
        log()
        ref_metrics = extract_tb_metrics(os.path.join(ref_dump, "tb"))
        resume_metrics = extract_tb_metrics(os.path.join(resume_dump, "tb"))
        compare_metrics(ref_metrics, resume_metrics, args.assert_equal)

    finally:
        if original:
            log()
            log(f"Restoring: {original}")
            subprocess.run(["git", "checkout", original], check=True)

    log(f"Logs saved in: {args.output_folder}")


if __name__ == "__main__":
    main()
