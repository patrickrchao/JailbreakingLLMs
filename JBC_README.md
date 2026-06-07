# JBC (JailbreakChat) baseline — PAIR Table 2

Reproduces the **JBC row** of PAIR Table 2. JBC is a *static-template* baseline:
each jailbreak template from jailbreakchat.com has its `[INSERT PROMPT HERE]`
placeholder replaced by the harmful objective and sent to the target **once**
(no iterative attacker). A behavior is **jailbroken if ANY template succeeds**.

It is **provider-agnostic and reusable** — point `--target-model` at any model
registered in `config.py` and it routes to the right provider/key automatically.
Same judge (`llama-guard-4-12b`) and dataset (JailbreakBench harmful) as the
PAIR Table-2 runs, so the numbers are directly comparable.

## Run

Export the API keys for your target + judge as environment variables, then call
`jbc_eval.py` directly:

```bash
export TOGETHER_API_KEY=...   # Qwen target + Llama-Guard-4 judge
export GEMINI_API_KEY=...     # only if targeting Gemini

python jbc_eval.py --target-model qwen3-235b-instruct --limit 50
python jbc_eval.py --target-model gemini-2.5-flash    --limit 50
python jbc_eval.py --target-model llama-2-7b-chat-hf  --limit 50   # different target
```

Output → `logs/jbc_<target>/status.jsonl` and a printed summary with the overall
**JB%** plus per-template effectiveness.

## For teammates (different target model)

1. Make sure your target is registered in `config.py` (model name → litellm name
   → API-key env var). The repo already has many; add yours if missing.
2. Export the needed API key(s) as environment variables (`TOGETHER_API_KEY`,
   `GEMINI_API_KEY`, `OPENAI_API_KEY` + `OPENAI_BASE_URL` for a gateway).
3. Run with `--target-model <yours>`. Everything else (templates, judge, dataset)
   is shared, so results are comparable across the team.

## Files
- `jbc_eval.py` — the evaluator (CLI: `--target-model`, `--judge-model`,
  `--limit`, `--start-index`, `--resume`, `--log-dir`).
- `jbc_templates.json` — the jailbreak templates (name, source, template). Each
  MUST contain the `[INSERT PROMPT HERE]` placeholder. Add/remove freely.

## Templates & faithfulness
- `AIM` is the exact template the PAIR paper published (Table 9) — the most
  popular jailbreakchat.com template.
- The others come from **TrustAIRLab/in-the-wild-jailbreak-prompts** (CCS'24),
  the canonical public archive of jailbreakchat.com prompts (the site is now
  defunct). Shipped set: AIM, DAN, STAN, DUDE, JailBreak, Burple, UnGpt,
  BetterDAN, evil (9 templates).
- ⚠️ The paper used **10** templates but only published the AIM one, so this is a
  faithful *approximation* of the JBC baseline, not the exact 10. Add a 10th to
  `jbc_templates.json` if you want — the tool is count-agnostic.
