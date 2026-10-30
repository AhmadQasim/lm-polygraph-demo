"""Time LM-Polygraph estimators on MPS at 7B, to see whether a full BioASQ run
is affordable on this machine.

Loads Qwen2.5-7B-Instruct in fp16 on MPS with eager attention (required: see
smoke_test_mps.py), then times MaximumSequenceProbability (logits only) and
SemanticEntropy (sampling + NLI clustering) over a handful of questions.

Run:  .venv/bin/python scripts/time_7b.py
"""

import argparse
import statistics
import time
import warnings

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from lm_polygraph import estimate_uncertainty
from lm_polygraph.utils.model import WhiteboxModel
from lm_polygraph.utils.generation_parameters import GenerationParameters

warnings.filterwarnings("ignore")

MODEL = "Qwen/Qwen2.5-7B-Instruct"

# Real BioASQ-style factoids: a mix the model should know and should not.
QUESTIONS = [
    "What is the mode of inheritance of Wilson's disease?",
    "Which protein is encoded by the gene GLT8D1?",
    "Orteronel was developed for treatment of which cancer?",
]

SYSTEM_HINT = (
    "Answer the biomedical question with a short phrase only. "
    "Do not explain. Question: "
)


def gb(n_bytes: float) -> str:
    return f"{n_bytes / 1024**3:.2f} GB"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--max-new-tokens", type=int, default=32)
    args = ap.parse_args()

    print(f"torch {torch.__version__} | device {args.device} | model {args.model}")

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16, attn_implementation="eager"
    ).to(args.device)
    hf_model.eval()
    load_s = time.time() - t0
    n_params = sum(p.numel() for p in hf_model.parameters())
    print(f"loaded in {load_s:.1f}s | {n_params/1e9:.2f}B params")
    if args.device == "mps":
        print(f"mps allocated after load: {gb(torch.mps.current_allocated_memory())}")

    model = WhiteboxModel(
        model=hf_model,
        tokenizer=tokenizer,
        model_path=args.model,
        generation_parameters=GenerationParameters(
            max_new_tokens=args.max_new_tokens,
            stop_strings=["\n"],
        ),
        instruct=True,
    )

    from lm_polygraph.estimators import MaximumSequenceProbability, SemanticEntropy

    results = {}
    for name, estimator in [
        ("MaximumSequenceProbability", MaximumSequenceProbability()),
        ("SemanticEntropy", SemanticEntropy()),
    ]:
        print(f"\n--- {name} ---")
        times = []
        for q in QUESTIONS:
            t = time.time()
            try:
                out = estimate_uncertainty(
                    model, estimator, input_text=SYSTEM_HINT + q
                )
                dt = time.time() - t
                times.append(dt)
                print(
                    f"  {dt:6.1f}s  u={out.uncertainty: 9.4f}  "
                    f"{out.generation_text.strip()[:55]!r}"
                )
            except Exception as exc:
                print(f"  FAILED after {time.time()-t:.1f}s: {type(exc).__name__}: {exc}")
                break
        if times:
            results[name] = statistics.median(times)
        if args.device == "mps":
            print(f"  peak mps allocated: {gb(torch.mps.driver_allocated_memory())}")

    print("\n=== extrapolation ===")
    for name, med in results.items():
        for n in (100, 250):
            total_min = med * n / 60
            print(f"  {name:28s} {n:4d} questions -> {total_min:7.1f} min")


if __name__ == "__main__":
    main()
