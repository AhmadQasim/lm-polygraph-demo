"""Worked examples for the three method slides: one right and one wrong answer, opened up.

The n=1000 caches store each estimator's final score only. The method slides show
what is inside those scores - the next-token distributions behind mean token
entropy, the verdict probabilities behind p(True), the ten samples behind lexical
similarity - so this script re-runs LM-Polygraph on the two example questions with
the run's own settings (Qwen2.5-3B-Instruct, bf16, greedy, stop at newline, 24 new
tokens) and saves those statistics to slides/method_examples.json, which is small
and tracked. The figures read only that file.

It checks itself against the GPU run: the greedy answers must be identical and
the two deterministic scores (mean token entropy, p(True)) must agree to within
MPS-vs-CUDA rounding. Lexical similarity is drawn fresh from 10 new samples, so
it is recorded next to the run's value rather than asserted equal.

Run:  HF_HUB_OFFLINE=1 .venv/bin/python scripts/method_examples.py   (~1 min on MPS)
"""

import json
import pathlib
import sys
import warnings

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
warnings.filterwarnings("ignore")

import bioasq  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
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
from lm_polygraph.stat_calculators.prompt import PromptCalculator  # noqa: E402
import lm_polygraph.estimators as E  # noqa: E402

MODEL = "Qwen/Qwen2.5-3B-Instruct"
CACHE = ROOT / "outputs" / "bioasq_Qwen2.5-3B-Instruct_n1000_seed0_t4.json"
OUT = ROOT / "slides" / "method_examples.json"
# Chosen because all three estimators rank them the right way round, and both are
# drug questions a biomedical audience reads at a glance. Matched by question text.
EXAMPLES = ["What type of drug is apixaban?",
            "Erenumab, used to treat migraine headaches, binds to what protein?"]
TOL = 0.05   # MPS against the CUDA run: bf16 rounding, not a different computation


def main() -> None:
    cache = json.loads(CACHE.read_text())
    idx = [cache["questions"].index(q) for q in EXAMPLES]
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    llm = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(device).eval()
    llm.generation_config.pad_token_id = tok.pad_token_id
    model = WhiteboxModel(model=llm, tokenizer=tok, model_path=MODEL,
                          generation_parameters=GenerationParameters(
                              max_new_tokens=24, stop_strings=["\n"]),
                          instruct=True)
    calcs = register_default_stat_calculators(
        model_type="Whitebox", output_attentions=False, output_hidden_states=False)
    torch.manual_seed(0)
    man = UEManager(
        data=Dataset(x=[bioasq.build_prompt(q) for q in EXAMPLES], y=["-"] * len(EXAMPLES),
                     batch_size=1),
        model=model,
        estimators=[E.MeanTokenEntropy(), E.PTrue(), E.LexicalSimilarity()],
        builder_env_stat_calc=BuilderEnvironmentStatCalculator(model),
        available_stat_calculators=calcs, generation_metrics=[], ue_metrics=[],
        processors=[], ignore_exceptions=False, verbose=False, max_new_tokens=24,
        save_stats=["greedy_texts", "greedy_tokens", "greedy_log_probs", "entropy",
                    "sample_texts", "p_true"])
    man()

    est = lambda name, j: float(man.estimations[("sequence", name)][j])  # noqa: E731
    lex = E.LexicalSimilarity()
    pc = PromptCalculator()
    out = []
    for j, i in enumerate(idx):
        answer = man.stats["greedy_texts"][j]
        assert answer == cache["generations"][i], (answer, cache["generations"][i])
        graded = bioasq.grade(answer, cache["golds"][i])
        bioasq.apply_manual_labels([graded], [EXAMPLES[j]], [answer], ROOT / "manual_labels.json")

        # --- mean token entropy: every step's distribution, as its top candidates
        steps = []
        for t, lp, h in zip(man.stats["greedy_tokens"][j], man.stats["greedy_log_probs"][j],
                            man.stats["entropy"][j]):
            p = np.exp(np.asarray(lp, dtype=np.float64))
            top = np.argsort(-p)[:4]
            steps.append({"token": tok.decode([t]), "entropy": float(h),
                          "top": [[tok.decode([int(k)]), float(p[k])] for k in top]})

        # --- p(True): the library's own prompt, and the verdict distribution
        text = pc.prompt.format(q=bioasq.build_prompt(EXAMPLES[j]), s="", a=answer)
        batch = {k: v.to(device) for k, v in model.tokenize([text]).items()}
        with torch.no_grad():
            lp = torch.log_softmax(llm(**batch).logits[0, -1].float(), -1)
        top = torch.topk(lp, 4)
        true_id = tok(["True"])["input_ids"][0][0]
        verdict = {"p_true": float(lp[true_id].exp()),
                   "top": [[tok.decode([int(k)]), float(v.exp())]
                           for v, k in zip(top.values, top.indices)]}
        # The stat the estimator reads must be this log-probability, not a logit.
        assert abs(float(lp[true_id]) - float(man.stats["p_true"][j])) < 1e-3

        # --- lexical similarity: the ten samples and their pairwise ROUGE-L
        samples = list(man.stats["sample_texts"][j])
        sims = [[lex._score_single(a, b) for b in samples] for a in samples]

        row = {
            "question": EXAMPLES[j], "golds": cache["golds"][i], "answer": answer,
            "correct": bool(graded["correct"]),
            "steps": steps, "verdict": verdict,
            "samples": samples, "pairwise_rougeL": sims,
            "scores": {n: est(n, j) for n in
                       ("MeanTokenEntropy", "PTrue", "LexicalSimilarity_rougeL")},
            "run_scores": {n: cache["scores"][n][i] for n in
                           ("MeanTokenEntropy", "PTrue", "LexicalSimilarity_rougeL")},
        }
        for n in ("MeanTokenEntropy", "PTrue"):
            assert abs(row["scores"][n] - row["run_scores"][n]) < TOL * max(1, abs(row["run_scores"][n])), n
        out.append(row)

    OUT.write_text(json.dumps({
        "_comment": "Written by scripts/method_examples.py; read by slides/fig_methods.py.",
        "model": MODEL, "device": device, "ptrue_prompt": pc.prompt,
        "examples": out}, indent=1, ensure_ascii=False) + "\n")
    for r in out:
        print(f"{r['answer']!r:18} correct={r['correct']}  now {r['scores']}  run {r['run_scores']}")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
