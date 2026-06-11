#!/usr/bin/env python3
"""Fast run: 60 instances to get valid output quickly."""
import asyncio, json, sys, os
sys.path.insert(0, '.')
os.environ.setdefault('OPENROUTER_API_KEY', os.environ.get('OPENROUTER_API_KEY',''))
import method as M

async def main():
    M.COST_CAP = 8.0
    instances = M.load_full_dataset()
    eval_instances = [i for i in instances if i.gold_answer not in ('unknown',)]
    # Take 20 per source for balanced 60-instance run
    by_src = {}
    for i in eval_instances:
        by_src.setdefault(i.source, []).append(i)
    subset = []
    for src, lst in by_src.items():
        subset.extend(lst[:20])
    subset = subset[:60]
    print(f"Running on {len(subset)} instances")

    stage0 = await M.stage0_pilot(subset[:15])
    print(f"Stage0: {stage0['decision']}")

    results = await M.run_experiment_v2(subset, max_concurrent=8)
    print(f"Got {len(results)} results | cost=${M._cumulative_cost:.4f}")

    metrics = M.compute_metrics(results)
    matched_cov = M.compute_matched_coverage_baselines(results)
    random_oracle = M.compute_random_oracle(results)
    strat = M.stratified_h2_analysis(results)
    div = M.atms_vs_problog_divergence(results)

    schema_out = M.format_result_for_schema(results)
    schema_out['metadata'] = {
        'method_name': 'Bounded-ATMS-v2',
        'model': M.CHEAP_MODEL,
        'n_instances': len(results),
        'n_instances_70b_ablation': 0,
        'stage0_tractability': stage0,
        'total_cost_usd': M._cumulative_cost,
        'parameters': {'depth_cap': M.DEPTH_CAP, 'beam_width': M.BEAM_WIDTH,
                       'linc_k': M.LINC_K},
        'evaluation': metrics,
        'matched_coverage_baselines': matched_cov,
        'random_oracle_accuracy': random_oracle,
        'assumption_load_stratification': strat,
        'atms_vs_problog_divergence': div,
        'model_ablation_results': {'model': M.LARGE_MODEL, 'n_instances': 0, 'metrics': {}},
        'note': 'fast_60_instance_run',
    }
    out = M.WORKSPACE / 'full_method_out.json'
    out.write_text(json.dumps(schema_out, indent=2, default=str))
    print(f"Saved {out} ({out.stat().st_size/1024:.1f} KB)")
    await M.close_session()

asyncio.run(main())
