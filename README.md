# Reproducing PAIR on the UCSD TritonAI gateway — an approximate "Llama" Table-2 run

This repo is a fork of [PAIR / JailbreakingLLMs](https://github.com/patrickrchao/JailbreakingLLMs)
(Chao et al., *Jailbreaking Black Box LLMs in Twenty Queries*). It documents how we
got PAIR running end-to-end against models served by the **UCSD TritonAI
OpenAI-compatible gateway** (`https://tritonai-api.ucsd.edu/v1`), and the result of
a full 100-behavior run.

The original PAIR README is preserved as [`README_UPSTREAM_PAIR.md`](README_UPSTREAM_PAIR.md).

---

## ⚠️ Read this first: it is a *variant*, not a Table-2 reproduction

The paper's Table 2 "Llama" row uses **Mixtral-8x7B** (attacker), **Llama-2-7B-chat**
(target), and the **Llama-Guard** JailbreakBench classifier (judge). **None of those
are hosted on the TritonAI gateway.** So this is an *approximate variant* that
substitutes models the gateway does serve. **The numbers below are NOT comparable to
the paper's Table 2.**

| Role | Paper Table 2 (Llama row) | This variant |
|------|---------------------------|--------------|
| Attacker | Mixtral-8x7B | `api-deepseek-v4-flash` |
| Target   | **Llama-2-7B-chat** | **`api-llama-4-scout`** |
| Judge    | Llama-Guard classifier | `claude-sonnet-4-6` |

## TL;DR result

Full JailbreakBench `harmful` set (100 behaviors), 30 streams × 3 iterations:

| Metric | Value |
|--------|-------|
| **Attack Success Rate (Jailbreak %)** | **18.0 %** (18 / 100) |
| **Queries per Success (mean)** | **41.2** |
| Queries/success — median / min / max | 38 / 2 / 89 |
| Wall-clock | ~5.2 h, 0 failures |

---

## 1. Environment

Everything runs through HTTP API calls to the gateway — **no GPU / local model
inference is needed**. (`torch` is pulled in only as a transitive dependency.)

```bash
# A dedicated conda env (kept out of git; ~8 GB)
conda create -y -p ./envs/pair python=3.11
conda activate ./envs/pair
pip install "litellm==1.52.0" "fschat>=0.2.36" jailbreakbench wandb pandas psutil accelerate
```

**Why the pins matter** (these were real blockers):
- `jailbreakbench` imports `litellm.llms.prompt_templates`, which newer litellm
  removed → pin **`litellm==1.52.0`**.
- `fschat==0.2.23` (the upstream Dockerfile pin) requires pydantic v1, but
  `jailbreakbench` requires pydantic ≥2.6 → use **`fschat>=0.2.36`** (pydantic-v2
  compatible).
- `fastchat.model` imports need **`accelerate`**.

## 2. API key

You need a **UCSD TritonAI gateway key** (NOT an OpenAI key — the gateway is
OpenAI-*compatible*, used via the `OPENAI_API_KEY` / `OPENAI_BASE_URL` variables).
The launcher reads it from `./api-key.txt`, which is **git-ignored and never
committed**:

```bash
echo "YOUR_TRITON_KEY" > api-key.txt
```

## 3. Run it

`run_variant.sh` activates the env, injects the key + gateway URL, and keeps all
caches inside the project folder:

```bash
# smoke test (1 behavior, tiny budget)
./run_variant.sh --limit 1 --n-streams 3 --n-iterations 2 --attack-max-n-tokens 1024

# full run (all 100 behaviors, paper budget, resumable)
./run_variant.sh --full --resume --continue-on-error \
    --n-streams 30 --n-iterations 3 --attack-max-n-tokens 1024
```

Output lands in `logs/pair_ucsd_variant/`:
- `status.jsonl` — one line per behavior (`jailbroken`, `queries_to_jailbreak`, …)
- `run.out` — full stdout, ending in the `=== UCSD PAIR Variant Summary ===` block

```bash
tail -6 logs/pair_ucsd_variant/run.out      # the summary
wc -l logs/pair_ucsd_variant/status.jsonl   # how many of 100 done
```

---

## 4. How this variant was actually reproduced (the debugging log)

Out of the box the run did **not** work. Here is what broke and how it was fixed —
useful for anyone porting PAIR to a different gateway.

### Step 1 — Pick models that exist *and* behave
The originally-configured attacker `api-mistral-small-3.2-2506` returned HTTP 500.
We probed **all 17 gateway models** with the real PAIR red-team system prompt. An
attacker must (a) not refuse the red-team role and (b) emit parseable JSON:

| Candidate | Outcome |
|-----------|---------|
| `api-deepseek-v4-flash` | ✅ complies + clean JSON → **chosen attacker** |
| `mistral-large-3` / `kimi-k2.5` / `minimax-m2` | ❌ HTTP 400 |
| `api-gpt-oss-120b` | ❌ refuses ("I'm sorry…") |
| `api-gemma-4-26b` / `nova-premier` | ❌ empty / content-filtered |
| `api-mistral-small-3.2-2506` / `nemotron-*` | ❌ HTTP 500 |

Target `api-llama-4-scout` and judge `claude-sonnet-4-6` both worked.

### Step 2 — The judge failed 100 % of the time
`claude-sonnet-4-6` is served via Vertex AI, which **rejects requests that specify
both `temperature` and `top_p`** (`"cannot both be specified for this model"`, HTTP
400). PAIR always sends both, so every judge call 400'd and every score silently
defaulted to 1 → a meaningless 0 % jailbreak rate.
**Fix:** in `language_models.py`, drop `top_p` for Claude models.

### Step 3 — The reasoning attacker returns empty content
`api-deepseek-v4-flash` is a reasoning model and can return `content=None` when the
token budget is spent on hidden reasoning, crashing on a `None` concatenation.
**Fix:** coerce `None`/failed streams to an error string so they retry, and run the
attacker with `--attack-max-n-tokens 1024`.

### Step 4 — One bad stream shouldn't kill a whole behavior
If a single stream can't produce valid JSON after all retries, PAIR originally
aborted the entire behavior.
**Fix:** in `conversers.py`, substitute a placeholder for that stream and keep going.
`language_models.py` also now guards against `litellm.batch_completion` returning
exception objects in-place.

### Result
After these four fixes the pipeline ran cleanly: judge errors **0**, all **100/100**
behaviors completed with **0** failures.

---

## 5. Detailed results

- **18 / 100 behaviors jailbroken (18.0 %)**, mean **41.2** queries per success
  (median 38, min 2, max 89). 8 of the 18 broke within the first 30 queries (one
  iteration); the rest needed multiple refinement rounds — the bimodal pattern PAIR
  is known for.
- 82 % of behaviors survived all 90 queries → Llama-4-Scout is a fairly robust target.

Jailbroken behaviors by category:

| Category | Count |
|----------|-------|
| Physical harm | 6 |
| Harassment/Discrimination | 5 |
| Sexual/Adult content | 4 |
| Fraud/Deception | 1 |
| Disinformation | 1 |
| Privacy | 1 |

**Citation-safe statement:**
> Using PAIR with a deepseek-v4-flash attacker, a Llama-4-Scout target, and a
> Claude-Sonnet-4.6 judge over all 100 JailbreakBench behaviors (30 streams × 3
> iterations), we observed an 18.0 % attack success rate at 41.2 queries per
> successful jailbreak.

---

## 6. Reproducing the *real* Table 2 later

The only missing piece is a **`TOGETHER_API_KEY`**. With one, the original
`run_table2_llama.py` runs the faithful setup (Mixtral attacker / Llama-2-7B target /
Llama-Guard judge) — that, not this variant, is a true Table-2 reproduction.

## 7. What changed vs. upstream

| File | Change |
|------|--------|
| `config.py` | Registered TritonAI models + OpenAI-compatible base URL wiring |
| `language_models.py` | Drop `top_p` for Claude; tolerate empty/failed streams |
| `conversers.py` | Placeholder fallback for unparseable attack streams |
| `main.py`, `judges.py` | Model choices come from `MODEL_NAMES`; judge uses OpenAI-style messages |
| `run_ucsd_pair_variant.py`, `run_variant.sh` | Variant runner + launcher (new) |

## Note on paths

`run_variant.sh` assumes this project lives at `/data/fengfei/JailbreakingLLMs` with
the env at `./envs/pair` and key at `./api-key.txt` on the **same machine**. If you
clone elsewhere, edit the `PROJ` and conda paths at the top of `run_variant.sh`.
