# PAIR — UCSD TritonAI variant (approximate "Llama" Table-2 setup)

This is a fork of [PAIR / JailbreakingLLMs](https://github.com/patrickrchao/JailbreakingLLMs)
adapted to run entirely through the **UCSD TritonAI OpenAI-compatible gateway**
(`https://tritonai-api.ucsd.edu/v1`).

> ⚠️ **This is NOT a strict reproduction of the paper's Table 2.** TritonAI does
> not host the original models (Llama-2-7B, Mixtral, Llama-Guard). It is an
> *approximate variant* that swaps in models the gateway does serve. The numbers
> here are **not comparable** to the paper's Table 2 numbers.

## Model setup (the working trio)

| Role | Model (TritonAI id) |
|------|---------------------|
| Attacker | `api-deepseek-v4-flash` |
| Target   | `api-llama-4-scout` |
| Judge    | `claude-sonnet-4-6` |

These were chosen by probing all 17 gateway models: deepseek is the only one that
both complies with the PAIR red-team system prompt **and** emits parseable JSON;
llama-4-scout is the target; claude-sonnet is the 1–10 judge.

## Latest result (100 JailbreakBench behaviors, 30 streams × 3 iterations)

| Metric | Value |
|--------|-------|
| Attack Success Rate (Jailbreak %) | **18.0 %** (18 / 100) |
| Queries per Success (mean) | **41.2** |
| Wall-clock | ~5.2 h, 0 failures |

## Prerequisites

- A **UCSD TritonAI gateway API key** (NOT an OpenAI key). The repo's
  `run_variant.sh` reads it from `./api-key.txt`. **This file is git-ignored and
  is never committed — place your own key there:**
  ```bash
  echo "YOUR_TRITON_KEY" > api-key.txt
  ```
- The Python environment. This fork was run on a shared machine with the env at
  `./envs/pair` (git-ignored). To recreate it anywhere:
  ```bash
  conda create -y -p ./envs/pair python=3.11
  conda activate ./envs/pair
  pip install "litellm==1.52.0" "fschat>=0.2.36" jailbreakbench wandb pandas psutil accelerate
  ```
  The pinned versions matter: `litellm==1.52.0` keeps the `prompt_templates` path
  that `jailbreakbench` imports, and `fschat>=0.2.36` is pydantic-v2 compatible
  (the old `fschat==0.2.23` conflicts with jailbreakbench).

## Run it

One-command launcher (activates the env, injects key + gateway URL, keeps all
caches inside the project folder):

```bash
# smoke test: 1 behavior, tiny budget
./run_variant.sh --limit 1 --n-streams 3 --n-iterations 2 --attack-max-n-tokens 1024

# full run: all 100 behaviors, paper budget, resumable
./run_variant.sh --full --resume --continue-on-error \
    --n-streams 30 --n-iterations 3 --attack-max-n-tokens 1024
```

Results land in `logs/pair_ucsd_variant/`:
- `status.jsonl` — one line per behavior (jailbroken?, queries-to-jailbreak, etc.)
- `run.out` — full stdout, ending in the `=== UCSD PAIR Variant Summary ===` block

Read the summary any time:
```bash
tail -6 logs/pair_ucsd_variant/run.out
wc -l logs/pair_ucsd_variant/status.jsonl   # how many of 100 done
```

### Useful flags
- `--limit N` / `--full` — number of behaviors (default 1).
- `--n-streams` / `--n-iterations` — PAIR budget (paper uses 30 × 3).
- `--attack-max-n-tokens 1024` — give the reasoning attacker room (deepseek can
  return empty content with fewer tokens).
- `--resume` — skip behaviors already completed in `status.jsonl`.
- `--continue-on-error` — keep going if one behavior crashes.
- `--dry-run` — validate config, make no API calls.

## What was changed vs. upstream PAIR

- `config.py` — registered the TritonAI models (`api-deepseek-v4-flash`,
  `api-llama-4-scout`, `claude-sonnet-4-6`, …) and wired them to the
  OpenAI-compatible base URL.
- `language_models.py` —
  (a) **drop `top_p` for Claude models** (Vertex-AI Claude rejects sending both
  `temperature` and `top_p`, which otherwise made the judge fail 100 %);
  (b) tolerate failed / empty (`content=None`) streams instead of crashing.
- `conversers.py` — if a stream can't produce valid attack JSON after all
  retries, substitute a placeholder instead of aborting the whole behavior.
- `main.py` / `judges.py` — model choices come from `MODEL_NAMES`; judge uses
  OpenAI-style messages.

## Note on paths

`run_variant.sh` assumes this project lives at
`/data/fengfei/JailbreakingLLMs` with the conda env at `./envs/pair` and the
key at `./api-key.txt` on the **same machine**. If you clone elsewhere, edit the
`PROJ` and conda paths at the top of `run_variant.sh`.

## Reproducing the *real* Table 2 later

Get a `TOGETHER_API_KEY` and use the original `run_table2_llama.py` (Mixtral
attacker / Llama-2-7B target / Llama-Guard judge) — that is the faithful setup.
