"""Run LM-Polygraph uncertainty estimators over BioASQ factoids and report AUROC.

Generations and scores are cached to outputs/ so the notebook (and a live demo)
can re-run instantly. Pass --force to recompute.

Run:  .venv/bin/python scripts/run_bioasq.py --n 100
"""

import argparse
import json
import pathlib
import sys
import time
import warnings

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import bioasq  # noqa: E402

from lm_polygraph.utils.model import WhiteboxModel  # noqa: E402
from lm_polygraph.utils.dataset import Dataset  # noqa: E402
from lm_polygraph.utils.manager import UEManager  # noqa: E402
from lm_polygraph.utils.generation_parameters import GenerationParameters  # noqa: E402
from lm_polygraph.utils.builder_enviroment_stat_calculator import (  # noqa: E402
    BuilderEnvironmentStatCalculator,
)
from lm_polygraph.defaults.register_default_stat_calculators import (  # noqa: E402
    register_default_stat_calculators,
)
import lm_polygraph.estimators as E  # noqa: E402

warnings.filterwarnings("ignore")

OUT = pathlib.Path(__file__).parent.parent / "outputs"


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"




TIERS = bioasq.TIERS


def estimators(max_tier: int):
    out = []
    for tier in sorted(TIERS):
        if tier > max_tier:
            break
        for name in TIERS[tier]:
            out.append(getattr(E, name)())
    return out


def run(args) -> dict:
    items = bioasq.load_factoid(n=args.n, seed=args.seed)
    prompts = [bioasq.build_prompt(it["question"]) for it in items]
    # Dataset wants a target string per input; we grade ourselves against the
    # full synonym set, so the first gold is only a placeholder here.
    targets = [it["golds"][0] for it in items]

    print(f"loading {args.model} ...")
    t0 = time.time()
    # padding_side="left" is REQUIRED for a decoder-only model whenever
    # batch_size > 1. lm-polygraph sets it inside the deprecated
    # WhiteboxModel.from_pretrained; building the model directly (as the
    # deprecation notice tells you to) skips that, and right-padding makes
    # batched generation silently wrong.
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    # attn_implementation matters: with output_attentions=True transformers 5.x
    # must use eager (sdpa returns no attention tensors and lm-polygraph's
    # greedy_probs indexes them unconditionally). We do not use any
    # attention-based estimator, so we turn attentions off and keep sdpa.
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation=args.attn
    ).to(args.device)
    hf_model.eval()
    print(f"  loaded in {time.time()-t0:.1f}s")

    model = WhiteboxModel(
        model=hf_model,
        tokenizer=tokenizer,
        model_path=args.model,
        generation_parameters=GenerationParameters(
            max_new_tokens=args.max_new_tokens, stop_strings=["\n"]
        ),
        instruct=True,
    )

    calcs = register_default_stat_calculators(
        model_type="Whitebox",
        output_attentions=False,
        output_hidden_states=False,
        deberta_batch_size=args.deberta_batch_size,
    )
    manager = UEManager(
        data=Dataset(x=prompts, y=targets, batch_size=args.batch_size),
        model=model,
        estimators=estimators(args.max_tier),
        builder_env_stat_calc=BuilderEnvironmentStatCalculator(model),
        available_stat_calculators=calcs,
        generation_metrics=[],
        ue_metrics=[],
        processors=[],
        ignore_exceptions=not args.strict,
        verbose=True,
        max_new_tokens=args.max_new_tokens,
        save_stats=["greedy_texts"],
    )

    t0 = time.time()
    manager()
    elapsed = time.time() - t0
    print(f"\nran {len(items)} questions in {elapsed/60:.1f} min")

    scores = {name: list(map(float, vals)) for (_lvl, name), vals in manager.estimations.items()}
    return {
        "model": args.model,
        "max_tier": args.max_tier,
        "n": len(items),
        "seed": args.seed,
        "elapsed_s": elapsed,
        # Record the full run config. Without it a cached result cannot be told
        # apart from one produced on different hardware or settings - which
        # matters once runs happen on more than one machine.
        "config": {
            "device": args.device,
            "dtype": "bfloat16",
            "attn": args.attn,
            "batch_size": args.batch_size,
            "max_new_tokens": args.max_new_tokens,
            "deberta_batch_size": args.deberta_batch_size,
            "torch": torch.__version__,
        },
        "questions": [it["question"] for it in items],
        "golds": [it["golds"] for it in items],
        "generations": list(manager.stats.get("greedy_texts", [])),
        "scores": scores,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--max-tier", type=int, default=4, choices=[1, 2, 3, 4],
                    help="cumulative cost tier; see TIERS")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=default_device())
    ap.add_argument("--attn", default="sdpa", choices=["sdpa", "eager"])
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--deberta-batch-size", type=int, default=10,
                    help="NLI batch size; raise well above the default on a CUDA GPU")
    ap.add_argument("--strict", action="store_true", help="do not swallow estimator errors")
    ap.add_argument("--force", action="store_true", help="recompute even if cached")
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    tag = args.model.split("/")[-1]
    cache = OUT / f"bioasq_{tag}_n{args.n}_seed{args.seed}_t{args.max_tier}.json"

    if cache.exists() and not args.force:
        print(f"using cached {cache}")
        result = json.loads(cache.read_text())
    else:
        result = run(args)
        cache.write_text(json.dumps(result, indent=2))
        print(f"cached -> {cache}")

    report(result)


def report(result: dict) -> None:
    from sklearn.metrics import roc_auc_score

    gens, golds = result["generations"], result["golds"]
    graded = [bioasq.grade(g, gg) for g, gg in zip(gens, golds)]
    correct = [g["correct"] for g in graded]
    n_corr = sum(correct)
    print(f"\naccuracy: {n_corr}/{len(correct)} = {n_corr/len(correct):.1%}")
    n_border = sum(g["borderline"] for g in graded)
    print(f"borderline gradings needing review: {n_border}")

    if n_corr in (0, len(correct)):
        print("!! only one class present - AUROC undefined")
        return

    # Positive class = the model was WRONG, so a good detector scores it high.
    y = [0 if c else 1 for c in correct]
    print(f"\n{'estimator':<34}{'AUROC':>8}")
    print("-" * 42)
    rows = []
    for name, vals in result["scores"].items():
        if len(vals) != len(y):
            continue
        try:
            rows.append((name, roc_auc_score(y, vals)))
        except ValueError:
            continue
    for name, auc in sorted(rows, key=lambda r: -r[1]):
        print(f"{name:<34}{auc:>8.3f}")


if __name__ == "__main__":
    main()
