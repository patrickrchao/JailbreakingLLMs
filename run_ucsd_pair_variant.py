"""Run a PAIR model-variant experiment through the UCSD TritonAI gateway."""

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

from config import MODEL_NAMES


DEFAULT_BASE_URL = "https://tritonai-api.ucsd.edu/v1"
DEFAULT_ATTACK_MODEL = "api-deepseek-v4-flash"
DEFAULT_TARGET_MODEL = "api-llama-4-scout"
DEFAULT_JUDGE_MODEL = "claude-sonnet-4-6"
FIRST_JAILBREAK_RE = re.compile(r"First Jailbreak: (\d+) Queries")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run PAIR with models served by the UCSD TritonAI OpenAI-compatible "
            "gateway. This is a model-variant experiment, not a strict Table 2 reproduction."
        )
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run all 100 harmful JailbreakBench behaviors.",
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
        help="Zero-based JailbreakBench dataset index to start from.",
    )
    parser.add_argument(
        "--attack-model",
        choices=MODEL_NAMES,
        default=DEFAULT_ATTACK_MODEL,
    )
    parser.add_argument(
        "--target-model",
        choices=MODEL_NAMES,
        default=DEFAULT_TARGET_MODEL,
    )
    parser.add_argument(
        "--judge-model",
        choices=MODEL_NAMES,
        default=DEFAULT_JUDGE_MODEL,
    )
    parser.add_argument("--n-streams", type=int, default=30)
    parser.add_argument("--n-iterations", type=int, default=3)
    parser.add_argument("--attack-max-n-tokens", type=int, default=500)
    parser.add_argument("--target-max-n-tokens", type=int, default=150)
    parser.add_argument(
        "--api-key-file",
        type=Path,
        default=Path.home() / "api-key.txt",
        help="Path to a plain-text UCSD API key. The key is never printed.",
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument(
        "--wandb-mode",
        choices=["online", "offline", "disabled"],
        default=os.environ.get("WANDB_MODE", "offline"),
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Directory for per-behavior logs and status.jsonl.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip behaviors already completed successfully in the status file.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue to later behaviors if one main.py process fails.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the experiment configuration without making API requests.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def scratch_root() -> Path | None:
    user = os.environ.get("USER")
    if not user:
        return None
    root = Path("/scratch") / user
    return root if root.exists() else None


def default_log_dir(args: argparse.Namespace) -> Path:
    experiment_name = "__".join(
        [slug(args.attack_model), slug(args.target_model), slug(args.judge_model)]
    )
    root = scratch_root()
    if root is not None:
        return root / "logs" / "pair_ucsd_variant" / experiment_name
    return Path("logs") / "pair_ucsd_variant" / experiment_name


def load_api_key(path: Path) -> str:
    expanded_path = path.expanduser()
    if expanded_path.exists():
        key = expanded_path.read_text(encoding="utf-8").strip()
        if key:
            return key
        raise ValueError(f"API key file is empty: {expanded_path}")

    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if key:
        return key
    raise FileNotFoundError(
        f"API key file not found: {expanded_path}. "
        "Pass --api-key-file or set OPENAI_API_KEY."
    )


def configure_environment(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env["OPENAI_API_KEY"] = load_api_key(args.api_key_file)
    env["OPENAI_BASE_URL"] = args.base_url
    env["WANDB_MODE"] = args.wandb_mode

    root = scratch_root()
    if root is not None:
        cache_root = root / "cache"
        env.setdefault("PIP_CACHE_DIR", str(cache_root / "pip"))
        env.setdefault("HF_HOME", str(cache_root / "huggingface"))
        env.setdefault(
            "TRANSFORMERS_CACHE", str(cache_root / "huggingface" / "transformers")
        )
        env.setdefault(
            "HF_DATASETS_CACHE", str(cache_root / "huggingface" / "datasets")
        )
        env.setdefault("WANDB_DIR", str(root / "wandb"))

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
    return env


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

    print("\n=== UCSD PAIR Variant Summary ===")
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
    dataset = jbb.read_dataset(split="harmful").as_dataframe()
    limit = len(dataset) if args.full else args.limit
    if args.start_index < 0 or args.start_index >= len(dataset):
        raise SystemExit(f"--start-index must be between 0 and {len(dataset) - 1}.")
    if limit < 1:
        raise SystemExit("--limit must be at least 1.")
    if args.n_streams < 1 or args.n_iterations < 1:
        raise SystemExit("--n-streams and --n-iterations must be at least 1.")

    stop_index = min(args.start_index + limit, len(dataset))
    selected_indices = set(range(args.start_index, stop_index))
    log_dir = args.log_dir or default_log_dir(args)
    status_path = log_dir / "status.jsonl"

    print("Experiment type: PAIR model variant, not strict Table 2 reproduction")
    print(f"Attacker: {args.attack_model}")
    print(f"Target: {args.target_model}")
    print(f"Judge: {args.judge_model}")
    print(f"Gateway: {args.base_url}")
    print(f"PAIR budget: {args.n_streams} streams x {args.n_iterations} iterations")
    print(f"Dataset rows: {args.start_index} through {stop_index - 1}")
    print(f"Logs: {log_dir}")
    if args.dry_run:
        print("Dry run complete. No API requests were made.")
        return 0

    env = configure_environment(args)
    log_dir.mkdir(parents=True, exist_ok=True)
    completed_indices = load_completed_indices(status_path) if args.resume else set()
    main_py = Path(__file__).resolve().parent / "main.py"

    for index in range(args.start_index, stop_index):
        if index in completed_indices:
            print(f"SKIP {index}: already completed")
            continue

        row = dataset.iloc[index]
        behavior = str(row["Behavior"])
        safe_behavior = slug(behavior)[:80]
        log_path = log_dir / f"{index:03d}_{safe_behavior}.log"
        command = [
            sys.executable,
            str(main_py),
            "--attack-model",
            args.attack_model,
            "--target-model",
            args.target_model,
            "--judge-model",
            args.judge_model,
            "--not-jailbreakbench",
            "--n-streams",
            str(args.n_streams),
            "--n-iterations",
            str(args.n_iterations),
            "--attack-max-n-tokens",
            str(args.attack_max_n_tokens),
            "--target-max-n-tokens",
            str(args.target_max_n_tokens),
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
            "attack_model": args.attack_model,
            "target_model": args.target_model,
            "judge_model": args.judge_model,
            "n_streams": args.n_streams,
            "n_iterations": args.n_iterations,
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
