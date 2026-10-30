"""Smoke test: does LM-Polygraph run on Apple Silicon (MPS)?

Loads a small instruct model, then runs one white-box estimator that needs only
logits (MaximumSequenceProbability) and one that additionally needs sampling and
an NLI model (SemanticEntropy). Prints a pass/fail line for each.

Run:  .venv/bin/python scripts/smoke_test_mps.py
"""

import argparse
import time
import warnings

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from lm_polygraph import estimate_uncertainty
from lm_polygraph.utils.model import WhiteboxModel
from lm_polygraph.utils.generation_parameters import GenerationParameters

warnings.filterwarnings("ignore")

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPTS = [
    # expected: model knows this, low uncertainty
    "Question: What is the mode of inheritance of Wilson's disease?\nAnswer:",
    # expected: niche, model should confabulate, high uncertainty
    "Question: Which protein is encoded by the gene GLT8D1?\nAnswer:",
]


def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--device", default="auto", choices=["auto", "mps", "cpu"])
    args = ap.parse_args()

    device = pick_device(args.device)
    print(f"torch {torch.__version__} | device: {device}")

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # attn_implementation="eager" is REQUIRED: with transformers 5.x the default
    # sdpa backend returns no attention tensors, and lm-polygraph's greedy_probs
    # stat calculator indexes into them unconditionally (IndexError otherwise).
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float32, attn_implementation="eager"
    ).to(device)
    hf_model.eval()
    print(f"loaded {args.model} in {time.time() - t0:.1f}s")

    model = WhiteboxModel(
        model=hf_model,
        tokenizer=tokenizer,
        model_path=args.model,
        generation_parameters=GenerationParameters(max_new_tokens=24),
    )

    # Imported here so an import error is attributed to the right stage.
    from lm_polygraph.estimators import MaximumSequenceProbability, SemanticEntropy

    for name, estimator in [
        ("MaximumSequenceProbability", MaximumSequenceProbability()),
        ("SemanticEntropy", SemanticEntropy()),
    ]:
        print(f"\n--- {name} ---")
        for prompt in PROMPTS:
            question = prompt.split("\n")[0].removeprefix("Question: ")
            t = time.time()
            try:
                out = estimate_uncertainty(model, estimator, input_text=prompt)
                print(
                    f"  [ok {time.time() - t:5.1f}s] uncertainty={out.uncertainty: .4f}"
                    f"  answer={out.generation_text.strip()[:60]!r}"
                )
                print(f"             q={question[:70]}")
            except Exception as exc:
                print(f"  [FAIL {time.time() - t:5.1f}s] {type(exc).__name__}: {exc}")
                print(f"             q={question[:70]}")


if __name__ == "__main__":
    main()
