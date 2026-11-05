"""The model and the estimators behind the UI: load once, then score one prompt at a time.

Settings follow scripts/run_bioasq.py, so a BioASQ-style question scored here is
comparable with the n=1000 runs: bf16, sdpa attention with attentions and hidden
states off (transformers 5.x breaks the attention path otherwise), left padding,
greedy decoding, LM-Polygraph's defaults for sampling (10 samples, temperature 1)
and NLI (microsoft/deberta-large-mnli).
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
import threading
import time
import warnings

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
warnings.filterwarnings("ignore")

import bioasq  # noqa: E402

OUTPUTS = ROOT / "outputs"
MANUAL = ROOT / "manual_labels.json"
HF_CACHE = pathlib.Path(os.environ.get("HF_HUB_CACHE")
                        or pathlib.Path(os.environ.get("HF_HOME", "~/.cache/huggingface"))
                        .expanduser() / "hub")

#: The models this project uses: the six n=1000 runs, plus the small model the
#: slide code is tested against. `run` names the cached BioASQ run, if there is one.
MODELS = [
    {"id": "Qwen/Qwen2.5-0.5B-Instruct", "label": "Qwen2.5 0.5B", "params_b": 0.5,
     "run": None, "note": "the slide-code test model; no BioASQ run"},
    {"id": "Qwen/Qwen2.5-3B-Instruct", "label": "Qwen2.5 3B", "params_b": 3.1,
     "run": "bioasq_Qwen2.5-3B-Instruct_n1000_seed0_t4.json"},
    {"id": "Qwen/Qwen2.5-7B-Instruct", "label": "Qwen2.5 7B", "params_b": 7.6,
     "run": "bioasq_Qwen2.5-7B-Instruct_n1000_seed0_t4.json"},
    {"id": "meta-llama/Llama-3.2-3B-Instruct", "label": "Llama 3.2 3B", "params_b": 3.2,
     "run": "bioasq_Llama-3.2-3B-Instruct_n1000_seed0_t4.json", "note": "gated on the Hub"},
    {"id": "meta-llama/Llama-3.1-8B-Instruct", "label": "Llama 3.1 8B", "params_b": 8.0,
     "run": "bioasq_Llama-3.1-8B-Instruct_n1000_seed0_t4.json", "note": "gated on the Hub"},
    {"id": "mistralai/Mistral-7B-Instruct-v0.3", "label": "Mistral 7B v0.3", "params_b": 7.2,
     "run": "bioasq_Mistral-7B-Instruct-v0.3_n1000_seed0_t4.json"},
    {"id": "allenai/OLMo-2-1124-7B-Instruct", "label": "OLMo 2 7B", "params_b": 7.3,
     "run": "bioasq_OLMo-2-1124-7B-Instruct_n1000_seed0_t4.json"},
]
MODEL_BY_ID = {m["id"]: m for m in MODELS}

#: Family of each estimator, as on the methods slides (fig_methods.TYPE_OF).
FAMILY = {
    "MaximumSequenceProbability": "Information", "Perplexity": "Information",
    "MeanTokenEntropy": "Information", "FisherRao": "Information", "RenyiNeg": "Information",
    "SelfCertainty": "Information", "BoostedProbSequence": "Information",
    "MeanPointwiseMutualInformation": "Information",
    "MeanConditionalPointwiseMutualInformation": "Information",
    "PTrue": "Self-report", "PTrueEmpirical": "Self-report",
    "LexicalSimilarity": "Consistency", "NumSemSets": "Consistency", "DegMat": "Consistency",
    "Eccentricity": "Consistency", "EigValLaplacian": "Consistency",
    "KernelLanguageEntropy": "Consistency",
    "MonteCarloSequenceEntropy": "Sampling + likelihood",
    "MonteCarloNormalizedSequenceEntropy": "Sampling + likelihood",
    "SemanticEntropy": "Sampling + likelihood",
}
PRETTY = {  # as plot_bioasq.PRETTY, keyed by the class name
    "MaximumSequenceProbability": "Max sequence prob.", "Perplexity": "Perplexity",
    "MeanTokenEntropy": "Mean token entropy", "FisherRao": "Fisher-Rao",
    "RenyiNeg": "Rényi-Neg", "SelfCertainty": "Self-certainty",
    "BoostedProbSequence": "Boosted prob.", "MeanPointwiseMutualInformation": "Mean PMI",
    "MeanConditionalPointwiseMutualInformation": "Mean cond. PMI", "PTrue": "p(True)",
    "PTrueEmpirical": "p(True) empirical", "LexicalSimilarity": "Lexical similarity",
    "NumSemSets": "NumSemSets", "DegMat": "DegMat", "Eccentricity": "Eccentricity",
    "EigValLaplacian": "EigValLaplacian", "KernelLanguageEntropy": "Kernel language entropy",
    "MonteCarloSequenceEntropy": "MC sequence entropy",
    "MonteCarloNormalizedSequenceEntropy": "MC norm. seq. entropy",
    "SemanticEntropy": "Semantic entropy",
}
TIER_NEEDS = {1: "no sampling", 2: "+ a second prompt", 3: "+ sampling", 4: "+ NLI model"}
TIER_OF = {n: t for t, names in bioasq.TIERS.items() for n in names}


def class_of(score_key: str) -> str:
    """'DegMat_NLI_score_entail' -> 'DegMat': the estimator's str() carries options."""
    return score_key.split("_")[0]


# =====================================================================================
# model availability
# =====================================================================================
def cached_weights(model_id: str) -> bool:
    """True if every weight file of the newest snapshot resolves to a real file.

    snapshot_download(local_files_only=True) is too strict here: the cache uses
    nested blob links, which it reports as an incomplete snapshot although
    from_pretrained loads them fine.
    """
    snaps = HF_CACHE / f"models--{model_id.replace('/', '--')}" / "snapshots"
    if not snaps.is_dir():
        return False
    for snap in sorted(snaps.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        weights = [p for p in snap.iterdir() if p.suffix in (".safetensors", ".bin")]
        if weights and all(p.resolve().is_file() for p in weights):
            return True
    return False


def model_list() -> list[dict]:
    out = []
    for m in MODELS:
        out.append({**m, "cached": cached_weights(m["id"]),
                    "has_run": bool(m["run"]) and (OUTPUTS / m["run"]).exists(),
                    "approx_gb": round(m["params_b"] * 2.0, 1)})   # bf16: 2 bytes/param
    return out


# =====================================================================================
# the BioASQ reference: where a score sits among the n=1000 run's answers
# =====================================================================================
class Reference:
    """Scores and right/wrong labels of one cached run, for context on a new score."""

    WINDOW = 100   # "the 100 BioASQ answers with the most similar score"

    def __init__(self, path: pathlib.Path):
        d = json.loads(path.read_text())
        graded = [bioasq.grade(g, gg) for g, gg in zip(d["generations"], d["golds"])]
        bioasq.apply_manual_labels(graded, d["questions"], d["generations"], MANUAL)
        self.wrong = np.array([0 if g["correct"] else 1 for g in graded])
        self.accuracy = float(1 - self.wrong.mean())
        self.golds = dict(zip(d["questions"], d["golds"]))
        self.n = len(self.wrong)
        self.by_key = {}
        for key, vals in d["scores"].items():
            s = np.asarray(vals, float)
            ok = np.isfinite(s)
            s, w = s[ok], self.wrong[ok]
            order = np.argsort(s, kind="stable")
            self.by_key[key] = {"sorted": s[order], "wrong_sorted": w[order],
                                "auroc": _auroc(w, s)}

    def context(self, key: str, score: float) -> dict | None:
        ref = self.by_key.get(key)
        if ref is None or not np.isfinite(score):
            return None
        s, w = ref["sorted"], ref["wrong_sorted"]
        # Ties sit in the middle of their run: p(True) empirical is often exactly 1.0,
        # and counting only the scores strictly below would call that a low score.
        left, right = np.searchsorted(s, score, "left"), np.searchsorted(s, score, "right")
        pos = (left + right) / 2
        lo = int(max(0, min(pos - self.WINDOW // 2, len(s) - self.WINDOW)))
        near = w[lo:lo + self.WINDOW]
        return {"percentile": float(pos / len(s)), "near_wrong": float(near.mean()),
                "near_n": int(len(near)), "auroc": ref["auroc"],
                "inverted": ref["auroc"] < 0.5}


def _auroc(wrong, score) -> float:
    """P(score of a wrong answer > score of a right one), ties counted half."""
    from scipy.stats import rankdata
    r = rankdata(score)
    n1, n0 = int(wrong.sum()), int(len(wrong) - wrong.sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((r[wrong == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# =====================================================================================
# the engine
# =====================================================================================
class Engine:
    """One loaded model at a time. Loading runs in a background thread."""

    def __init__(self):
        self.lock = threading.Lock()
        self.state = "idle"          # idle | loading | ready | error
        self.message = "no model loaded"
        self.model_id = None
        self.device = None
        self.model = None
        self.tok = None
        self.calcs = None
        self.env = None
        self.reference = None
        self.loaded_s = None

    # ---- status -------------------------------------------------------------------
    def status(self) -> dict:
        m = MODEL_BY_ID.get(self.model_id) or {}
        return {"state": self.state, "message": self.message, "model": self.model_id,
                "label": m.get("label"), "device": self.device,
                "reference": None if self.reference is None else {
                    "run": m.get("run"), "n": self.reference.n,
                    "accuracy": self.reference.accuracy},
                "loaded_s": self.loaded_s}

    # ---- loading ------------------------------------------------------------------
    def start_load(self, model_id: str) -> None:
        if model_id not in MODEL_BY_ID:
            raise ValueError(f"unknown model {model_id!r}")
        if self.state == "loading":
            raise RuntimeError("a model is already loading")
        self.state, self.message, self.model_id = "loading", f"loading {model_id} …", model_id
        threading.Thread(target=self._load, args=(model_id,), daemon=True).start()

    def _load(self, model_id: str) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from lm_polygraph.utils.model import WhiteboxModel
        from lm_polygraph.utils.generation_parameters import GenerationParameters
        from lm_polygraph.utils.builder_enviroment_stat_calculator import (
            BuilderEnvironmentStatCalculator)
        from lm_polygraph.defaults.register_default_stat_calculators import (
            register_default_stat_calculators)
        t0 = time.time()
        try:
            with self.lock:
                self._unload()
                device = ("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
                local = cached_weights(model_id)
                self.message = (f"loading {model_id} from the local cache …" if local else
                                f"downloading {model_id} (~{MODEL_BY_ID[model_id]['params_b'] * 2:.0f} GB) …")
                tok = AutoTokenizer.from_pretrained(model_id, padding_side="left",
                                                    local_files_only=local)
                if tok.pad_token is None:      # Mistral ships none (see run_bioasq.py)
                    tok.pad_token = tok.eos_token
                llm = AutoModelForCausalLM.from_pretrained(
                    model_id, dtype=torch.bfloat16, attn_implementation="sdpa",
                    local_files_only=local).to(device).eval()
                if llm.generation_config.pad_token_id is None:
                    llm.generation_config.pad_token_id = tok.pad_token_id
                self.model = WhiteboxModel(
                    model=llm, tokenizer=tok, model_path=model_id,
                    generation_parameters=GenerationParameters(max_new_tokens=24,
                                                               stop_strings=["\n"]),
                    instruct=True)
                self.tok = tok
                self.calcs = register_default_stat_calculators(
                    model_type="Whitebox", output_attentions=False,
                    output_hidden_states=False, deberta_batch_size=16)
                self.env = BuilderEnvironmentStatCalculator(self.model)
                run = MODEL_BY_ID[model_id]["run"]
                self.message = "reading the BioASQ reference run …"
                self.reference = (Reference(OUTPUTS / run)
                                  if run and (OUTPUTS / run).exists() else None)
                self.device = device
                self.loaded_s = round(time.time() - t0, 1)
                self.state, self.message = "ready", f"{model_id} ready on {device}"
        except Exception as e:  # surfaced to the UI as is
            self._unload()
            self.state, self.message = "error", f"{type(e).__name__}: {e}"

    def _unload(self) -> None:
        import gc
        import torch
        self.model = self.tok = self.calcs = self.env = self.reference = None
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- one prompt ---------------------------------------------------------------
    def run(self, text: str, max_tier: int = 2, bioasq_prompt: bool = True,
            max_new_tokens: int | None = None) -> dict:
        import torch
        import lm_polygraph.estimators as E
        from lm_polygraph.utils.dataset import Dataset
        from lm_polygraph.utils.generation_parameters import GenerationParameters
        from lm_polygraph.utils.manager import UEManager

        if self.state != "ready":
            raise RuntimeError("no model is ready")
        if not text.strip():
            raise ValueError("empty prompt")
        max_tier = int(max(1, min(4, max_tier)))
        if max_new_tokens is None:
            max_new_tokens = 24 if bioasq_prompt else 128
        max_new_tokens = int(max(1, min(512, max_new_tokens)))

        with self.lock:
            t0 = time.time()
            prompt = bioasq.build_prompt(text.strip()) if bioasq_prompt else text
            # BioASQ mode answers in one line, as in the runs; free mode may run on.
            self.model.generation_parameters = GenerationParameters(
                max_new_tokens=max_new_tokens,
                stop_strings=["\n"] if bioasq_prompt else [])
            estimators = [getattr(E, n)() for t in sorted(bioasq.TIERS) if t <= max_tier
                          for n in bioasq.TIERS[t]]
            stats = ["greedy_texts", "greedy_tokens", "greedy_log_probs", "entropy"]
            if max_tier >= 3:
                stats.append("sample_texts")
            man = UEManager(
                data=Dataset(x=[prompt], y=[""], batch_size=1), model=self.model,
                estimators=estimators, builder_env_stat_calc=self.env,
                available_stat_calculators=self.calcs, generation_metrics=[],
                ue_metrics=[], processors=[], ignore_exceptions=True, verbose=False,
                max_new_tokens=max_new_tokens, save_stats=stats)
            man()
            elapsed = time.time() - t0

            answer = man.stats["greedy_texts"][0]
            tokens = []
            for t, lp, h in zip(man.stats["greedy_tokens"][0],
                                man.stats["greedy_log_probs"][0],
                                man.stats["entropy"][0]):
                p = np.exp(np.asarray(lp, dtype=np.float64))
                top = np.argsort(-p)[:5]
                tokens.append({"token": self.tok.decode([int(t)]), "entropy": float(h),
                               "p": float(p[int(t)]),
                               "top": [[self.tok.decode([int(k)]), float(p[k])] for k in top]})

            done = {name for (_lvl, name) in man.estimations}
            rows = []
            for est in estimators:
                key = str(est)
                cls = class_of(key)
                vals = man.estimations.get(("sequence", key))
                score = float(vals[0]) if vals is not None and len(vals) else float("nan")
                ok = key in done and np.isfinite(score)
                ctx = (self.reference.context(key, score)
                       if ok and self.reference is not None and bioasq_prompt else None)
                rows.append({"key": key, "name": PRETTY.get(cls, cls),
                             "family": FAMILY.get(cls, "?"), "tier": TIER_OF.get(cls),
                             "score": score if ok else None, "failed": not ok,
                             "reference": ctx})

            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            return {
                "answer": answer, "prompt_mode": "bioasq" if bioasq_prompt else "free",
                "max_tier": max_tier, "max_new_tokens": max_new_tokens,
                "elapsed_s": round(elapsed, 2), "tokens": tokens,
                "samples": list(man.stats.get("sample_texts", [[]])[0]) if max_tier >= 3 else [],
                "estimators": rows,
                "graded": self._grade(text.strip(), answer) if bioasq_prompt else None,
            }

    def _grade(self, question: str, answer: str) -> dict | None:
        """Right or wrong, when the prompt is one of the 1,000 BioASQ questions."""
        golds = QUESTION_BANK.golds.get(question)
        if golds is None:
            return None
        g = bioasq.grade(answer, golds)
        bioasq.apply_manual_labels([g], [question], [answer], MANUAL)
        return {"correct": bool(g["correct"]), "golds": golds[:4],
                "borderline": bool(g.get("borderline"))}


class QuestionBank:
    """The 1,000 BioASQ questions of the runs (the same for every model), for the
    "random question" button and for grading. Read from any cached run."""

    def __init__(self):
        self.golds, self.questions = {}, []
        for m in MODELS:
            if m["run"] and (OUTPUTS / m["run"]).exists():
                d = json.loads((OUTPUTS / m["run"]).read_text())
                self.questions = d["questions"]
                self.golds = dict(zip(d["questions"], d["golds"]))
                break

    def random(self) -> dict | None:
        if not self.questions:
            return None
        q = self.questions[np.random.randint(len(self.questions))]
        return {"question": q, "golds": self.golds[q][:4]}


QUESTION_BANK = QuestionBank()
