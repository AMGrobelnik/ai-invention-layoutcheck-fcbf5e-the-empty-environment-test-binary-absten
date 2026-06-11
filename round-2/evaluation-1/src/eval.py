#!/usr/bin/env python3
"""Rigorous statistical evaluation of ATMS Hallucination Meter — Iter 1 results."""

import gc
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from scipy import stats
from statsmodels.stats.proportion import proportion_confint

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")

WORKSPACE = Path(__file__).parent
LOG_DIR = WORKSPACE / "logs"
LOG_DIR.mkdir(exist_ok=True)
logger.add(LOG_DIR / "eval.log", rotation="30 MB", level="DEBUG")

FIGURES_DIR = WORKSPACE / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

DATA_PATH = Path(
    "/home/adrian/projects/ai-inventor/aii_data/users/admin/runs"
    "/run_r4ZXzXG-rGek/3_invention_loop/iter_1/gen_art"
    "/gen_art_experiment_1/full_method_out.json"
)


# ── helpers ──────────────────────────────────────────────────────────────────

def wilson_ci(count: int, nobs: int, alpha: float = 0.05) -> tuple[float, float]:
    if nobs == 0:
        return (0.0, 0.0)
    lo, hi = proportion_confint(count, nobs, alpha=alpha, method="wilson")
    return float(lo), float(hi)


def bootstrap_arc(
    errors: np.ndarray,
    scores: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Return (auc, ci_lower, ci_upper) for the risk-coverage curve via bootstrap."""
    n = len(errors)
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        e_b = errors[idx]
        s_b = scores[idx]
        order = np.argsort(s_b)
        e_sorted = e_b[order]
        risks = np.cumsum(e_sorted) / np.arange(1, n + 1)
        aucs.append(float(np.mean(risks)))
    order_full = np.argsort(scores)
    e_sorted_full = errors[order_full]
    risks_full = np.cumsum(e_sorted_full) / np.arange(1, n + 1)
    auc = float(np.mean(risks_full))
    ci_lo = float(np.percentile(aucs, 2.5))
    ci_hi = float(np.percentile(aucs, 97.5))
    return auc, ci_lo, ci_hi


def auroc_bootstrap(
    y_true: np.ndarray,
    scores: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """AUROC with bootstrap CI (manual computation, no sklearn needed for sklearn-free fallback)."""
    from sklearn.metrics import roc_auc_score  # already installed

    n = len(y_true)
    rng = np.random.default_rng(seed)

    if len(np.unique(y_true)) < 2:
        logger.warning("AUROC degenerate (single class) — returning 0.5")
        return 0.5, 0.5, 0.5

    auc = float(roc_auc_score(y_true, scores))
    boot_aucs = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        y_b = y_true[idx]
        s_b = scores[idx]
        if len(np.unique(y_b)) < 2:
            continue
        boot_aucs.append(float(roc_auc_score(y_b, s_b)))
    if not boot_aucs:
        return auc, auc, auc
    return auc, float(np.percentile(boot_aucs, 2.5)), float(np.percentile(boot_aucs, 97.5))


# ── data loading ──────────────────────────────────────────────────────────────

@logger.catch(reraise=True)
def load_data(path: Path) -> list[dict]:
    logger.info(f"Loading {path}")
    raw = json.loads(path.read_text())
    records = []
    for ds_block in raw["datasets"]:
        ds_name = ds_block["dataset"]
        for ex in ds_block["examples"]:
            r = dict(ex)
            r["dataset"] = ds_name
            # parse assumption_load
            al = r.get("metadata_assumption_load", "0")
            try:
                r["assumption_load"] = np.inf if str(al).lower() == "inf" else float(al)
            except (ValueError, TypeError):
                logger.warning(f"Unparseable assumption_load={al!r}, skipping example")
                continue
            # boolean strings
            r["empty_env_derivable"] = str(r.get("metadata_empty_env_derivable", "False")) == "True"
            r["l1_grounding_verified"] = str(r.get("metadata_l1_grounding_verified", "False")) == "True"
            # correct flags
            gold = r.get("output", "")
            r["correct_our_method"] = r.get("predict_our_method", "") == gold
            r["correct_cot"] = r.get("predict_cot", "") == gold
            r["correct_linc"] = r.get("predict_linc", "") == gold
            r["correct_problog"] = r.get("predict_problog", "") == gold
            # abstain: assumption_load == inf
            r["abstained"] = np.isinf(r["assumption_load"])
            records.append(r)
    logger.info(f"Loaded {len(records)} instances from {len(raw['datasets'])} datasets")
    return records


# ── Section 2: proportions ────────────────────────────────────────────────────

def compute_proportions(records: list[dict]) -> list[dict]:
    n = len(records)
    rows = []

    def add(metric, condition_fn, correct_fn):
        subset = [r for r in records if condition_fn(r)]
        count = sum(1 for r in subset if correct_fn(r))
        nobs = len(subset)
        p = count / nobs if nobs else 0.0
        lo, hi = wilson_ci(count, nobs)
        rows.append({"metric": metric, "count": count, "nobs": nobs,
                     "proportion": round(p, 6), "ci_lower": round(lo, 6), "ci_upper": round(hi, 6)})

    # empty-env precision for our method
    add("empty_env_precision_our_method",
        lambda r: r["empty_env_derivable"],
        lambda r: r["correct_our_method"])

    # overall accuracy (answered only)
    add("accuracy_answered_our_method",
        lambda r: not r["abstained"],
        lambda r: r["correct_our_method"])

    # overall accuracy all 48
    add("accuracy_all_our_method",
        lambda r: True,
        lambda r: r["correct_our_method"])

    # abstain rate
    abstain_count = sum(1 for r in records if r["abstained"])
    p_abs = abstain_count / n
    lo, hi = wilson_ci(abstain_count, n)
    rows.append({"metric": "abstain_rate", "count": abstain_count, "nobs": n,
                 "proportion": round(p_abs, 6), "ci_lower": round(lo, 6), "ci_upper": round(hi, 6)})

    # l1 grounding rate
    l1_count = sum(1 for r in records if r["l1_grounding_verified"])
    lo, hi = wilson_ci(l1_count, n)
    rows.append({"metric": "l1_grounding_rate", "count": l1_count, "nobs": n,
                 "proportion": round(l1_count / n, 6), "ci_lower": round(lo, 6), "ci_upper": round(hi, 6)})

    # baselines (all 48, no abstentions)
    for m in ["cot", "linc", "problog"]:
        add(f"accuracy_all_{m}",
            lambda r: True,
            lambda r, m=m: r[f"correct_{m}"])

    return rows


# ── Section 3: risk-coverage ──────────────────────────────────────────────────

def compute_risk_coverage(records: list[dict]) -> tuple[dict, dict]:
    n = len(records)
    # our_method: score = assumption_load (inf → 1e9)
    load_scores = np.array([min(r["assumption_load"], 1e9) for r in records])
    errors_ours = np.array([0.0 if r["correct_our_method"] else 1.0 for r in records])

    auc_our, ci_lo_our, ci_hi_our = bootstrap_arc(errors_ours, load_scores)

    # baselines: constant score → flat line at error rate
    arc_results = {"our_method": {"auc": round(auc_our, 6),
                                   "ci_lower": round(ci_lo_our, 6),
                                   "ci_upper": round(ci_hi_our, 6)}}
    flat_arcs = {}
    for m in ["cot", "linc", "problog"]:
        errors_b = np.array([0.0 if r[f"correct_{m}"] else 1.0 for r in records])
        flat_err = float(errors_b.mean())
        flat_arcs[m] = flat_err
        arc_results[m] = {"auc": round(flat_err, 6),
                          "ci_lower": round(flat_err, 6),
                          "ci_upper": round(flat_err, 6),
                          "note": "Baselines lack per-instance confidence; ARC = flat error rate"}

    # curve data for plotting
    order = np.argsort(load_scores)
    e_sorted = errors_ours[order]
    coverages = np.arange(1, n + 1) / n
    risks = np.cumsum(e_sorted) / np.arange(1, n + 1)

    return arc_results, {"coverages": coverages, "risks": risks, "flat_arcs": flat_arcs,
                         "errors_ours": errors_ours, "load_scores": load_scores}


def plot_risk_coverage(arc_data: dict, n: int):
    fig, ax = plt.subplots(figsize=(7, 5))
    coverages = arc_data["coverages"]
    risks = arc_data["risks"]
    flat_arcs = arc_data["flat_arcs"]
    errors_ours = arc_data["errors_ours"]
    load_scores = arc_data["load_scores"]

    # bootstrap band
    rng = np.random.default_rng(42)
    boot_risks = []
    for _ in range(1000):
        idx = rng.choice(n, size=n, replace=True)
        e_b = errors_ours[idx]
        s_b = load_scores[idx]
        order_b = np.argsort(s_b)
        boot_risks.append(np.cumsum(e_b[order_b]) / np.arange(1, n + 1))
    boot_risks = np.array(boot_risks)
    lo_band = np.percentile(boot_risks, 2.5, axis=0)
    hi_band = np.percentile(boot_risks, 97.5, axis=0)

    ax.plot(coverages, risks, color="steelblue", linewidth=2, label="ATMS (ours)")
    ax.fill_between(coverages, lo_band, hi_band, alpha=0.25, color="steelblue")

    colors = {"cot": "darkorange", "linc": "green", "problog": "red"}
    for m, err in flat_arcs.items():
        ax.axhline(err, linestyle="--", color=colors[m], linewidth=1.5, label=f"{m.upper()} (flat={err:.3f})")

    ax.set_xlabel("Coverage", fontsize=12)
    ax.set_ylabel("Selective Risk (error rate)", fontsize=12)
    ax.set_title(f"Risk-Coverage Curves (n={n})", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    path = str(FIGURES_DIR / "risk_coverage.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved {path}")
    return path


# ── Section 4: CWA ───────────────────────────────────────────────────────────

def compute_cwa(records: list[dict], coverage_targets: list[float]) -> list[dict]:
    n = len(records)
    load_scores = [min(r["assumption_load"], 1e9) for r in records]
    order = sorted(range(n), key=lambda i: load_scores[i])

    rows = []
    for c in coverage_targets:
        k = max(1, math.floor(c * n))
        subset_idx = order[:k]
        subset = [records[i] for i in subset_idx]
        for method in ["our_method", "cot", "linc", "problog"]:
            if method == "our_method":
                correct = sum(1 for r in subset if r["correct_our_method"])
            else:
                # baselines answer all → CWA at any coverage = sample accuracy
                # for consistent comparison, use the same k instances
                correct = sum(1 for r in subset if r[f"correct_{method}"])
            acc = correct / k
            lo, hi = wilson_ci(correct, k)
            rows.append({"method": method, "coverage_target": round(c, 4), "n_answered": k,
                         "accuracy": round(acc, 6), "ci_lower": round(lo, 6), "ci_upper": round(hi, 6)})
    return rows


def plot_cwa(cwa_table: list[dict]):
    fig, ax = plt.subplots(figsize=(7, 5))
    methods = ["our_method", "cot", "linc", "problog"]
    colors = {"our_method": "steelblue", "cot": "darkorange", "linc": "green", "problog": "red"}
    labels = {"our_method": "ATMS (ours)", "cot": "CoT", "linc": "LINC", "problog": "ProbLog"}

    for m in methods:
        rows = [r for r in cwa_table if r["method"] == m]
        xs = [r["coverage_target"] for r in rows]
        ys = [r["accuracy"] for r in rows]
        ax.plot(xs, ys, marker="o", color=colors[m], linewidth=1.8, label=labels[m])

    ax.set_xlabel("Coverage Target", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Coverage-Weighted Accuracy vs Coverage Level", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    path = str(FIGURES_DIR / "cwa_vs_coverage.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved {path}")
    return path


# ── Section 5: assumption-load vs error ──────────────────────────────────────

def compute_load_error_analysis(records: list[dict]) -> dict:
    sentinel = 10.0
    load_clipped = np.array([sentinel if np.isinf(r["assumption_load"]) else r["assumption_load"]
                              for r in records])
    errors = np.array([0.0 if r["correct_our_method"] else 1.0 for r in records])

    rho, pval = stats.spearmanr(load_clipped, errors)

    auroc, ci_lo, ci_hi = auroc_bootstrap(errors, load_clipped)

    return {
        "spearman_rho": round(float(rho), 6),
        "spearman_p": round(float(pval), 6),
        "auroc_assumption_load": {
            "auroc": round(auroc, 6),
            "ci_lower": round(ci_lo, 6),
            "ci_upper": round(ci_hi, 6),
        },
    }


def compute_calibration(records: list[dict]) -> list[dict]:
    strata = {
        "load=0 (empty-env)": lambda r: r["assumption_load"] == 0.0,
        "load=1": lambda r: r["assumption_load"] == 1.0,
        "load=inf": lambda r: np.isinf(r["assumption_load"]),
    }
    rows = []
    for label, fn in strata.items():
        subset = [r for r in records if fn(r)]
        n = len(subset)
        n_correct = sum(1 for r in subset if r["correct_our_method"])
        frac = n_correct / n if n else 0.0
        lo, hi = wilson_ci(n_correct, n)
        underpowered = n < 5
        rows.append({"stratum": label, "n": n, "n_correct": n_correct,
                     "fraction_correct": round(frac, 6),
                     "ci_lower": round(lo, 6), "ci_upper": round(hi, 6),
                     "underpowered": underpowered})
        if underpowered:
            logger.warning(f"Stratum '{label}' underpowered (n={n} < 5)")
    return rows


def plot_calibration(cal_strata: list[dict]):
    fig, ax = plt.subplots(figsize=(7, 5))
    labels = [r["stratum"] for r in cal_strata]
    fracs = [r["fraction_correct"] for r in cal_strata]
    los = [r["fraction_correct"] - r["ci_lower"] for r in cal_strata]
    his = [r["ci_upper"] - r["fraction_correct"] for r in cal_strata]
    ns = [r["n"] for r in cal_strata]

    x = np.arange(len(labels))
    bars = ax.bar(x, fracs, color=["steelblue", "darkorange", "tomato"], alpha=0.8,
                  yerr=[los, his], capsize=5, error_kw={"linewidth": 1.5})

    for bar, n_val in zip(bars, ns):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
                f"n={n_val}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Fraction Correct", fontsize=12)
    ax.set_title("Accuracy by Assumption-Load Stratum", fontsize=13)
    ax.set_ylim(0, 1.15)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    path = str(FIGURES_DIR / "calibration_by_load.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved {path}")
    return path


# ── Section 6: L1 degradation theorem ────────────────────────────────────────

def l1_degradation_theorem(records: list[dict]) -> dict:
    observed_eps = 0.0 if all(r["l1_grounding_verified"] for r in records) else (
        sum(1 for r in records if not r["l1_grounding_verified"]) / len(records)
    )
    return {
        "statement": (
            "Let epsilon_L1 be the per-atom L1 grounding error rate (probability any given atom "
            "is incorrectly grounded), with grounding errors independent across atoms. "
            "For an empty-environment derivation requiring a chain of k atoms: "
            "P(derivation fully correct) >= (1 - epsilon_L1)^k. "
            "Therefore empty-environment precision >= (1 - epsilon_L1)^k_max, "
            "where k_max is the maximum chain length in the derivation."
        ),
        "union_bound": "precision_EE >= 1 - k * epsilon_L1",
        "jensen_bound": "E[precision_EE] >= exp(-epsilon_L1 * k_bar) for mean depth k_bar",
        "instantiation_04_k3": round((1 - 0.04) ** 3, 6),
        "instantiation_04_k3_union_bound": round(1 - 3 * 0.04, 6),
        "observed_epsilon_l1": round(observed_eps, 6),
        "note": (
            "With observed epsilon_L1=0.0 (all instances L1-verified), the bound collapses to 1.0. "
            "The bound is non-trivial for epsilon_L1 > 0: e.g., epsilon_L1=0.04 (Path-of-Thoughts "
            "95.9% triplet accuracy), k=3 hops → lower bound = (0.96)^3 = 0.885."
        ),
    }


# ── Section 7: ATMS vs ProbLog divergence ─────────────────────────────────────

def compute_divergence(records: list[dict]) -> tuple[list[dict], int, int]:
    divergent = [r for r in records if r.get("predict_our_method") != r.get("predict_problog")]
    logger.info(f"Found {len(divergent)} ATMS-vs-ProbLog divergent instances")
    rows = []
    for r in divergent:
        gold = r.get("output", "")
        atms_pred = r.get("predict_our_method", "")
        prob_pred = r.get("predict_problog", "")
        rows.append({
            "dataset": r["dataset"],
            "input_snippet": r.get("input", "")[:100],
            "gold": gold,
            "predict_atms": atms_pred,
            "predict_problog": prob_pred,
            "atms_correct": atms_pred == gold,
            "problog_correct": prob_pred == gold,
        })
    atms_correct_count = sum(1 for r in rows if r["atms_correct"])
    problog_correct_count = sum(1 for r in rows if r["problog_correct"])
    return rows, atms_correct_count, problog_correct_count


# ── Additional metrics ────────────────────────────────────────────────────────

def compute_per_dataset_metrics(records: list[dict]) -> dict:
    """Accuracy per dataset per method."""
    result = {}
    datasets = sorted({r["dataset"] for r in records})
    for ds in datasets:
        sub = [r for r in records if r["dataset"] == ds]
        n = len(sub)
        result[ds] = {}
        for m in ["our_method", "cot", "linc", "problog"]:
            if m == "our_method":
                correct = sum(1 for r in sub if r["correct_our_method"])
            else:
                correct = sum(1 for r in sub if r[f"correct_{m}"])
            result[ds][f"accuracy_{m}"] = round(correct / n, 6)
            result[ds]["n"] = n
    return result


# ── Output assembly ───────────────────────────────────────────────────────────

@logger.catch(reraise=True)
def main():
    records = load_data(DATA_PATH)
    n = len(records)

    logger.info("Step 2: Computing proportions with Wilson CIs")
    proportions_ci = compute_proportions(records)

    logger.info("Step 3: Computing risk-coverage curves and AUC")
    arc_results, arc_data = compute_risk_coverage(records)

    coverage_targets = [0.10, 0.229, 0.50, 1.00]
    logger.info("Step 4: Computing coverage-weighted accuracy")
    cwa_table = compute_cwa(records, coverage_targets)

    logger.info("Step 5: Assumption-load vs error analysis")
    load_analysis = compute_load_error_analysis(records)
    cal_strata = compute_calibration(records)

    logger.info("Step 6: L1 degradation theorem")
    l1_theorem = l1_degradation_theorem(records)

    logger.info("Step 7: ATMS-vs-ProbLog divergence")
    divergence_rows, atms_div_correct, problog_div_correct = compute_divergence(records)

    logger.info("Generating figures")
    fig_rc = plot_risk_coverage(arc_data, n)
    fig_cal = plot_calibration(cal_strata)
    fig_cwa = plot_cwa(cwa_table)

    logger.info("Assembling eval_out.json")
    per_ds = compute_per_dataset_metrics(records)

    # flat aggregate metrics for schema compliance
    metrics_agg = {
        "n_instances": n,
        "empty_env_precision": proportions_ci[0]["proportion"],
        "accuracy_answered_our_method": proportions_ci[1]["proportion"],
        "accuracy_all_our_method": proportions_ci[2]["proportion"],
        "abstain_rate": proportions_ci[3]["proportion"],
        "l1_grounding_rate": proportions_ci[4]["proportion"],
        "accuracy_all_cot": proportions_ci[5]["proportion"],
        "accuracy_all_linc": proportions_ci[6]["proportion"],
        "accuracy_all_problog": proportions_ci[7]["proportion"],
        "arc_our_method": arc_results["our_method"]["auc"],
        "arc_cot": arc_results["cot"]["auc"],
        "arc_linc": arc_results["linc"]["auc"],
        "arc_problog": arc_results["problog"]["auc"],
        "spearman_rho": load_analysis["spearman_rho"],
        "spearman_p": load_analysis["spearman_p"],
        "auroc_assumption_load": load_analysis["auroc_assumption_load"]["auroc"],
        "atms_correct_on_divergent": float(atms_div_correct),
        "problog_correct_on_divergent": float(problog_div_correct),
        "n_divergent_instances": float(len(divergence_rows)),
        "observed_epsilon_l1": l1_theorem["observed_epsilon_l1"],
        "l1_bound_eps04_k3": l1_theorem["instantiation_04_k3"],
    }

    # rich metadata (all structured results)
    metadata = {
        "evaluation_name": "ATMS Hallucination Meter Iter-1 Rigorous Evaluation",
        "n_instances": n,
        "proportions_ci": proportions_ci,
        "arc_results": arc_results,
        "cwa_table": cwa_table,
        "spearman_rho": load_analysis["spearman_rho"],
        "spearman_p": load_analysis["spearman_p"],
        "auroc_assumption_load": load_analysis["auroc_assumption_load"],
        "calibration_by_load_stratum": cal_strata,
        "l1_degradation_theorem": l1_theorem,
        "atms_problog_divergence": divergence_rows,
        "atms_correct_on_divergent": atms_div_correct,
        "problog_correct_on_divergent": problog_div_correct,
        "per_dataset_metrics": per_ds,
        "figure_paths": [fig_rc, fig_cal, fig_cwa],
    }

    # per-example eval annotations
    datasets_out = {}
    for r in records:
        ds = r["dataset"]
        if ds not in datasets_out:
            datasets_out[ds] = []

        al = r["assumption_load"]
        eval_assumption_load_clipped = float(min(al, 10.0))

        example_out = {
            "input": r.get("input", ""),
            "output": r.get("output", ""),
            "predict_our_method": r.get("predict_our_method", ""),
            "predict_cot": r.get("predict_cot", ""),
            "predict_linc": r.get("predict_linc", ""),
            "predict_problog": r.get("predict_problog", ""),
            "metadata_hop_count": r.get("metadata_hop_count"),
            "metadata_assumption_load": r.get("metadata_assumption_load"),
            "metadata_empty_env_derivable": r.get("metadata_empty_env_derivable"),
            "metadata_n_choice_points": r.get("metadata_n_choice_points"),
            "metadata_cumulative_cost_usd": r.get("metadata_cumulative_cost_usd"),
            "metadata_wall_clock_sec": r.get("metadata_wall_clock_sec"),
            "metadata_l1_grounding_verified": r.get("metadata_l1_grounding_verified"),
            "eval_correct_our_method": float(r["correct_our_method"]),
            "eval_correct_cot": float(r["correct_cot"]),
            "eval_correct_linc": float(r["correct_linc"]),
            "eval_correct_problog": float(r["correct_problog"]),
            "eval_abstained": float(r["abstained"]),
            "eval_assumption_load_clipped": eval_assumption_load_clipped,
        }
        datasets_out[ds].append(example_out)

    datasets_list = [{"dataset": ds, "examples": examples}
                     for ds, examples in datasets_out.items()]

    eval_out = {
        "metadata": metadata,
        "metrics_agg": metrics_agg,
        "datasets": datasets_list,
    }

    out_path = WORKSPACE / "eval_out.json"
    out_path.write_text(json.dumps(eval_out, indent=2))
    logger.info(f"Wrote {out_path}")

    # summary
    logger.info("=" * 60)
    logger.info(f"n_instances: {n}")
    logger.info(f"empty_env_precision: {proportions_ci[0]['proportion']:.3f} "
                f"[{proportions_ci[0]['ci_lower']:.3f}, {proportions_ci[0]['ci_upper']:.3f}]")
    logger.info(f"accuracy_answered: {proportions_ci[1]['proportion']:.3f}")
    logger.info(f"abstain_rate: {proportions_ci[3]['proportion']:.3f}")
    logger.info(f"arc_our_method: {arc_results['our_method']['auc']:.4f} "
                f"[{arc_results['our_method']['ci_lower']:.4f}, {arc_results['our_method']['ci_upper']:.4f}]")
    logger.info(f"spearman_rho={load_analysis['spearman_rho']:.4f} p={load_analysis['spearman_p']:.4f}")
    logger.info(f"auroc_load={load_analysis['auroc_assumption_load']['auroc']:.4f}")
    logger.info(f"ATMS vs ProbLog divergent: {len(divergence_rows)}, "
                f"ATMS correct={atms_div_correct}, ProbLog correct={problog_div_correct}")


if __name__ == "__main__":
    main()
