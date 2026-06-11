#!/usr/bin/env python3
"""Mini test: 3 instances from dataset."""
import asyncio
import json
import sys
sys.path.insert(0, '.')
import method as M

async def main():
    M.COST_CAP = 2.0  # tight budget for mini test
    instances = M.load_full_dataset()
    # Take 1 per source for smoke test (3 total)
    by_src = {}
    for inst in instances:
        if inst.source not in by_src and inst.gold_answer != 'unknown':
            by_src[inst.source] = inst
        if len(by_src) == 3:
            break
    mini = list(by_src.values())
    print(f"Running mini test on {len(mini)} instances")
    results = await M.run_experiment_v2(mini, max_concurrent=3)
    print(f"Got {len(results)} results, cost=${M._cumulative_cost:.5f}")
    for r in results:
        print(f"  {r['source']} | load={r['assumption_load']} | our={r['our_method_eval']} | gold={r['gold']} | cot={r['cot']}")
    await M.close_session()
    print("MINI TEST PASSED")

asyncio.run(main())
