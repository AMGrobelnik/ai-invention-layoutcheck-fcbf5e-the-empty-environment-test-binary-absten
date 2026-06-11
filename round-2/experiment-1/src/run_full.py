#!/usr/bin/env python3
"""Full experiment runner."""
import asyncio
import os
import sys
import time

sys.path.insert(0, '.')
import method as M

@M.logger.catch(reraise=True)
async def main():
    t_start = time.perf_counter()
    M.logger.info("=== STARTING FULL EXPERIMENT ===")

    instances = M.load_full_dataset()
    eval_insts = [i for i in instances if i.gold_answer != 'unknown']
    M.logger.info(f"Eval instances (non-unknown): {len(eval_insts)}")

    # Stage 0
    stage0 = await M.stage0_pilot(eval_insts[:15])
    M.logger.info(f"Stage0: {stage0}")

    if stage0['decision'] == 'tractability_failure':
        M.logger.error("Tractability failure, aborting")
        await M.close_session()
        return

    # Full experiment
    results = await M.run_experiment_v2(eval_insts, max_concurrent=6)
    M.logger.info(f"Main experiment done: {len(results)} results | cost=${M._cumulative_cost:.4f}")

    # 70B ablation if budget allows
    results_70b = []
    budget_left = M.COST_CAP - M._cumulative_cost
    M.logger.info(f"Budget remaining: ${budget_left:.2f}")
    if budget_left > 3.0:
        M.logger.info("Starting 70B ablation on 150 instances")
        results_70b = await M.run_70b_ablation(eval_insts, n_subset=150)
        M.logger.info(f"70B ablation done: {len(results_70b)} results | cost=${M._cumulative_cost:.4f}")
    else:
        M.logger.warning(f"Skipping 70B ablation: budget too low ${budget_left:.2f}")

    # Compute all analyses
    metrics = M.compute_metrics(results)
    matched_cov = M.compute_matched_coverage_baselines(results)
    random_oracle = M.compute_random_oracle(results)
    strat = M.stratified_h2_analysis(results)
    div = M.atms_vs_problog_divergence(results)
    metrics_70b = M.compute_metrics(results_70b) if results_70b else {}

    M.logger.info(f"Spearman rho={metrics['spearman_rho']}")
    M.logger.info(f"EE precision={metrics['empty_env_precision']}")
    M.logger.info(f"Overall acc ATMS={metrics['overall_accuracy']['our_method']}")

    # Build schema output
    schema_out = M.format_result_for_schema(results)
    schema_out["metadata"] = {
        "method_name": "Bounded-ATMS-v2",
        "description": "Iter-2: matched-coverage baselines, 70B ablation, stratification, ATMS-vs-ProbLog ground-truth",
        "model": M.CHEAP_MODEL,
        "n_instances": len(results),
        "n_instances_70b_ablation": len(results_70b),
        "stage0_tractability": stage0,
        "total_cost_usd": M._cumulative_cost,
        "total_wall_clock_sec": time.perf_counter() - t_start,
        "parameters": {
            "depth_cap": M.DEPTH_CAP,
            "beam_width": M.BEAM_WIDTH,
            "assumption_threshold": M.ASSUMPTION_THRESHOLD,
            "linc_k": M.LINC_K,
        },
        "evaluation": metrics,
        "matched_coverage_baselines": matched_cov,
        "random_oracle_accuracy": random_oracle,
        "assumption_load_stratification": strat,
        "atms_vs_problog_divergence": {
            "n_total": div["n_total"],
            "n_divergent": div["n_divergent"],
            "divergence_rate": div["divergence_rate"],
            "atms_wins_on_divergent": div["atms_wins_on_divergent"],
            "problog_wins_on_divergent": div["problog_wins_on_divergent"],
            "both_wrong_on_divergent": div["both_wrong_on_divergent"],
        },
        "model_ablation_results": {
            "model": M.LARGE_MODEL,
            "n_instances": len(results_70b),
            "metrics": metrics_70b,
        },
    }

    out_path = M.WORKSPACE / "full_method_out.json"
    import json
    out_path.write_text(json.dumps(schema_out, indent=2, default=str))
    size_mb = out_path.stat().st_size / 1024 / 1024
    M.logger.info(f"Saved: {out_path} ({size_mb:.2f} MB)")

    # Round-trip check
    try:
        json.loads(out_path.read_text())
        M.logger.info("JSON round-trip: PASSED")
    except Exception as e:
        M.logger.error(f"JSON round-trip FAILED: {e}")

    await M.close_session()
    M.logger.info(f"=== DONE total_cost=${M._cumulative_cost:.4f} elapsed={time.perf_counter()-t_start:.0f}s ===")

asyncio.run(main())
