"""Time LM-Polygraph estimators on MPS at 7B, to see whether a full BioASQ run
is affordable on this machine.

Notes on the settings here, all of which were learned the hard way:
  * attn_implementation="eager" is REQUIRED. transformers 5.x defaults to sdpa,
    which returns no attention tensors, and lm-polygraph's greedy_probs
    calculator indexes into them unconditionally (IndexError otherwise).
  * dtype defaults to bfloat16, the dtype Qwen2.5 was trained in. An earlier
    float16 run produced '!' (token id 0) spliced into answers; bf16 is the
    safe default regardless of whether fp16 was the true cause.
  * The first estimator call pays MPS kernel compilation. It is run once as a
    warmup and excluded from the reported timings.

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

QUESTIONS = [
    "What is the mode of inheritance of Wilson's disease?",
    "Which protein is encoded by the gene GLT8D1?",
    "Orteronel was developed for treatment of which cancer?",
    "Name a synonym of Acrokeratosis paraneoplastica.",
    "Where is the protein Pannexin1 located?",
]
WARMUP_Q = "What is the capital of France?"

PREFIX = (
    "Answer the biomedical question with a short phrase only. "
    "Do not explain. Question: "
)

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def gb(n: float) -> str:
    return f"{n / 1024**3:.2f} GB"


def swap_mb() -> float:
    import subprocess

    out = subprocess.run(
        ["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True
    ).stdout
    # e.g. "total = 11264.00M  used = 9815.75M  free = 1448.25M"
    for part in out.split():
        if part.endswith("M") and "used" in out.split(part)[0][-8:]:
            return float(part[:-1])
    return float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--dtype", default="bf16", choices=list(DTYPES))
    ap.add_argument("--max-new-tokens", type=int, default=20)
    args = ap.parse_args()

    print(f"torch {torch.__version__} | mps | {args.model} | dtype={args.dtype}")
    print(f"swap used before load: {swap_mb():.0f} MB")

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=DTYPES[args.dtype], attn_implementation="eager"
    ).to("mps")
    hf_model.eval()
    print(f"loaded in {time.time() - t0:.1f}s | {sum(p.numel() for p in hf_model.parameters())/1e9:.2f}B params")
    print(f"mps allocated: {gb(torch.mps.current_allocated_memory())} | swap used: {swap_mb():.0f} MB")

    model = WhiteboxModel(
        model=hf_model,
        tokenizer=tokenizer,
        model_path=args.model,
        generation_parameters=GenerationParameters(
            max_new_tokens=args.max_new_tokens, stop_strings=["\n"]
        ),
        instruct=True,
    )

    from lm_polygraph.estimators import MaximumSequenceProbability, SemanticEntropy

    summary = {}
    for name, estimator in [
        ("MaximumSequenceProbability", MaximumSequenceProbability()),
        ("SemanticEntropy", SemanticEntropy()),
    ]:
        print(f"\n--- {name} ---")
        t = time.time()
        estimate_uncertainty(model, estimator, input_text=PREFIX + WARMUP_Q)
        print(f"  warmup (excluded): {time.time() - t:.1f}s")

        times = []
        for q in QUESTIONS:
            t = time.time()
            try:
                out = estimate_uncertainty(model, estimator, input_text=PREFIX + q)
                dt = time.time() - t
                times.append(dt)
                print(f"  {dt:6.1f}s  u={out.uncertainty: 9.4f}  {out.generation_text.strip()[:50]!r}")
            except Exception as exc:
                print(f"  FAILED {time.time()-t:.1f}s {type(exc).__name__}: {exc}")
                break
        if times:
            summary[name] = (statistics.median(times), min(times), max(times))
        print(f"  peak mps: {gb(torch.mps.driver_allocated_memory())} | swap used: {swap_mb():.0f} MB")

    print("\n=== per-call spread (warmup excluded) ===")
    for name, (med, lo, hi) in summary.items():
        print(f"  {name:28s} median {med:6.1f}s  min {lo:6.1f}s  max {hi:6.1f}s")
    print("\n=== extrapolation (median) ===")
    for name, (med, _, _) in summary.items():
        for n in (100, 250):
            print(f"  {name:28s} {n:4d} q -> {med*n/60:7.1f} min")


if __name__ == "__main__":
    main()
