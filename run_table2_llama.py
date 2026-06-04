"""Run the PAIR portion of Table 2 against the Llama-2 target model."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import jailbreakbench as jbb


FIRST_JAILBREAK_RE = re.compile(r"First Jailbreak: (\d+) Queries")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run PAIR with Mixtral as the attacker, Llama-2 as the target, and "
            "the JailbreakBench classifier as the judge."
        )
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run all 100 harmful JailbreakBench behaviors using the test phase.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1,
        help="Number of behaviors to run when --full is not set. Defaults to 1.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Zero-based dataset index to start from.",
    )
    parser.add_argument(
        "--phase",
        choices=["dev", "test", "eval"],
        default="dev",
        help="JailbreakBench logging phase. --full always uses test.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip behaviors that already completed successfully in the status file.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue to later behaviors if one main.py process fails.",
    )
    parser.add_argument(
        "--wandb-mode",
        choices=["online", "offline", "disabled"],
        default=os.environ.get("WANDB_MODE", "offline"),
        help="W&B mode used by main.py. Defaults to WANDB_MODE or offline.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Directory for per-behavior logs and status.jsonl.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_scratch_root() -> Path | None:
    user = os.environ.get("USER")
    if not user:
        return None
    root = Path("/scratch") / user
    return root if root.exists() else None


def configure_environment(wandb_mode: str) -> tuple[dict[str, str], Path]:
    env = os.environ.copy()
    scratch_root = default_scratch_root()

    if scratch_root is not None:
        cache_root = scratch_root / "cache"
        env.setdefault("PIP_CACHE_DIR", str(cache_root / "pip"))
        env.setdefault("HF_HOME", str(cache_root / "huggingface"))
        env.setdefault(
            "TRANSFORMERS_CACHE", str(cache_root / "huggingface" / "transformers")
        )
        env.setdefault(
            "HF_DATASETS_CACHE", str(cache_root / "huggingface" / "datasets")
        )
        env.setdefault("WANDB_DIR", str(scratch_root / "wandb"))
        default_log_dir = scratch_root / "logs" / "pair_table2_llama"
    else:
        default_log_dir = Path("logs") / "pair_table2_llama"

    env["WANDB_MODE"] = wandb_mode
    for variable in [
        "PIP_CACHE_DIR",
        "HF_HOME",
        "TRANSFORMERS_CACHE",
        "HF_DATASETS_CACHE",
        "WANDB_DIR",
    ]:
        value = env.get(variable)
        if value:
            Path(value).mkdir(parents=True, exist_ok=True)

    return env, default_log_dir


def load_completed_indices(status_path: Path) -> set[int]:
    completed: set[int] = set()
    if not status_path.exists():
        return completed

    with status_path.open() as status_file:
        for line in status_file:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("returncode") == 0:
                completed.add(int(record["index"]))
    return completed


def append_status(status_path: Path, record: dict) -> None:
    with status_path.open("a") as status_file:
        status_file.write(json.dumps(record, ensure_ascii=True) + "\n")


def run_and_tee(command: list[str], env: dict[str, str], log_path: Path) -> int:
    with log_path.open("w") as log_file:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
            log_file.flush()
        return process.wait()


def parse_jailbreak_result(log_path: Path) -> tuple[bool | None, int | None]:
    text = log_path.read_text(errors="replace")
    match = FIRST_JAILBREAK_RE.search(text)
    if match:
        return True, int(match.group(1))
    if "No jailbreaks achieved." in text:
        return False, None
    return None, None


def latest_records(status_path: Path) -> dict[int, dict]:
    records: dict[int, dict] = {}
    if not status_path.exists():
        return records
    with status_path.open() as status_file:
        for line in status_file:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            records[int(record["index"])] = record
    return records


def print_summary(status_path: Path, selected_indices: set[int]) -> None:
    records = latest_records(status_path)
    selected_records = [
        records[index]
        for index in sorted(selected_indices)
        if index in records and records[index].get("returncode") == 0
    ]
    jailbroken = [record for record in selected_records if record.get("jailbroken") is True]
    queries = [
        record["queries_to_jailbreak"]
        for record in jailbroken
        if record.get("queries_to_jailbreak") is not None
    ]

    print("\n=== Table 2 Llama-2 PAIR Summary ===")
    print(f"Completed behaviors: {len(selected_records)}/{len(selected_indices)}")
    if selected_records:
        rate = 100 * len(jailbroken) / len(selected_records)
        print(f"Jailbreak %: {rate:.1f}% ({len(jailbroken)}/{len(selected_records)})")
    if queries:
        print(f"Queries per Success: {sum(queries) / len(queries):.1f}")
    else:
        print("Queries per Success: no successful jailbreaks")
    print(f"Status file: {status_path}")


def main() -> int:
    args = parse_args()
    if "TOGETHER_API_KEY" not in os.environ:
        raise SystemExit("Missing TOGETHER_API_KEY. Export it before running this script.")

    env, default_log_dir = configure_environment(args.wandb_mode)
    log_dir = args.log_dir or default_log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    status_path = log_dir / "status.jsonl"

    dataset = jbb.read_dataset(split="harmful").as_dataframe()
    limit = len(dataset) if args.full else args.limit
    phase = "test" if args.full else args.phase
    if args.start_index < 0 or args.start_index >= len(dataset):
        raise SystemExit(f"--start-index must be between 0 and {len(dataset) - 1}.")
    if limit < 1:
        raise SystemExit("--limit must be at least 1.")

    stop_index = min(args.start_index + limit, len(dataset))
    selected_indices = set(range(args.start_index, stop_index))
    completed_indices = load_completed_indices(status_path) if args.resume else set()
    main_py = Path(__file__).resolve().parent / "main.py"

    print(f"Dataset rows: {args.start_index} through {stop_index - 1}")
    print(f"JailbreakBench phase: {phase}")
    print(f"W&B mode: {args.wandb_mode}")
    print(f"Logs: {log_dir}")

    for index in range(args.start_index, stop_index):
        if index in completed_indices:
            print(f"SKIP {index}: already completed")
            continue

        row = dataset.iloc[index]
        behavior = str(row["Behavior"])
        safe_behavior = re.sub(r"[^A-Za-z0-9._-]+", "_", behavior).strip("_")[:80]
        log_path = log_dir / f"{index:03d}_{safe_behavior}.log"
        command = [
            sys.executable,
            str(main_py),
            "--attack-model",
            "mixtral",
            "--target-model",
            "llama-2-7b-chat-hf",
            "--judge-model",
            "jailbreakbench",
            "--n-streams",
            "30",
            "--n-iterations",
            "3",
            "--attack-max-n-tokens",
            "500",
            "--target-max-n-tokens",
            "150",
            "--jailbreakbench-phase",
            phase,
            "--goal",
            str(row["Goal"]),
            "--target-str",
            str(row["Target"]),
            "--category",
            behavior,
            "--index",
            str(index),
            "-v",
        ]

        print(f"\nRUN {index}: {behavior}")
        started_at = utc_now()
        returncode = run_and_tee(command, env, log_path)
        jailbroken, queries_to_jailbreak = parse_jailbreak_result(log_path)
        record = {
            "index": index,
            "behavior": behavior,
            "category": str(row["Category"]),
            "phase": phase,
            "returncode": returncode,
            "jailbroken": jailbroken,
            "queries_to_jailbreak": queries_to_jailbreak,
            "started_at": started_at,
            "finished_at": utc_now(),
            "log_file": str(log_path),
        }
        append_status(status_path, record)

        if returncode != 0:
            print(f"FAILED {index}: return code {returncode}")
            if not args.continue_on_error:
                print_summary(status_path, selected_indices)
                return returncode

    print_summary(status_path, selected_indices)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
