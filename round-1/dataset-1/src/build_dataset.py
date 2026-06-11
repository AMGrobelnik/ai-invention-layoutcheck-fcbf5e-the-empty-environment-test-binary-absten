#!/usr/bin/env python3
"""Build unified ATMS benchmark dataset: CLUTRR + bAbI + RuleTaker + ProofWriter."""

import ast
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from loguru import logger

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add("logs/run.log", rotation="30 MB", level="DEBUG")

WS = Path(__file__).parent
DS = WS / "temp" / "datasets"
random.seed(42)


def load_json(path: Path) -> list:
    return json.loads(path.read_text())


@logger.catch(reraise=True)
def process_clutrr(rows: list, target: int = 20) -> list:
    """Sample target instances per hop level from CLUTRR test split."""
    by_hop: dict[int, list] = defaultdict(list)
    for r in rows:
        hop = len(ast.literal_eval(r["edge_types"]))
        by_hop[hop].append(r)

    out = []
    for hop in sorted(by_hop):
        pool = by_hop[hop]
        sample = random.sample(pool, min(target, len(pool)))
        for i, r in enumerate(sample):
            names = ast.literal_eval(r["query"])
            query = f"What is the relationship between {names[0]} and {names[1]}?"
            out.append({
                "id": f"CLUTRR_test_{r['id']}",
                "domain": "kinship",
                "document_text": r["story"],
                "query": query,
                "gold_answer": r["target_text"],
                "hop_count": hop,
                "source_dataset": "CLUTRR",
                "split": "test",
            })
    logger.info(f"CLUTRR: {len(out)} instances, hops {sorted(by_hop.keys())}")
    return out


@logger.catch(reraise=True)
def process_babi(rows: list, tasks: list[int] = [2, 3, 15], per_task: int = 33) -> list:
    """Sample per_task instances per bAbI task from tasks list."""
    task_domain = {2: "causal", 3: "causal", 15: "deduction"}
    task_hops = {2: 2, 3: 3, 15: 1}

    by_task: dict[int, list] = defaultdict(list)
    for r in rows:
        t = r["task"]
        if t in tasks:
            by_task[t].append(r)

    out = []
    for t in tasks:
        pool = by_task[t]
        sample = random.sample(pool, min(per_task, len(pool)))
        for i, r in enumerate(sample):
            out.append({
                "id": f"bAbI_qa{t}_test_{i}",
                "domain": task_domain[t],
                "document_text": r["passage"],
                "query": r["question"],
                "gold_answer": r["answer"],
                "hop_count": task_hops[t],
                "source_dataset": f"bAbI_qa{t}",
                "split": "test",
            })
        logger.info(f"bAbI task {t}: {len(sample)} instances")
    return out


@logger.catch(reraise=True)
def process_ruletaker(rows: list, depths: list[str] = ["depth-0", "depth-1", "depth-2", "depth-3"], per_depth: int = 25) -> list:
    """Sample per_depth instances per depth from RuleTaker."""
    depth_to_hop = {"depth-0": 0, "depth-1": 1, "depth-2": 2, "depth-3": 3}

    by_depth: dict[str, list] = defaultdict(list)
    for r in rows:
        if r["config"] in depths:
            by_depth[r["config"]].append(r)

    out = []
    for d in depths:
        pool = by_depth[d]
        sample = random.sample(pool, min(per_depth, len(pool)))
        for i, r in enumerate(sample):
            # entailment → True, not entailment → False
            gold = "True" if r["label"] == "entailment" else "False"
            out.append({
                "id": f"RuleTaker_{d}_{i}",
                "domain": "explicit_rule",
                "document_text": r["context"],
                "query": r["question"],
                "gold_answer": gold,
                "hop_count": depth_to_hop[d],
                "source_dataset": "RuleTaker",
                "split": "train",
            })
        logger.info(f"RuleTaker {d}: {len(sample)} instances")
    return out


@logger.catch(reraise=True)
def process_proofwriter(rows: list, max_qdep: int = 3, per_qdep: int = 25) -> list:
    """Sample per_qdep instances per QDep from ProofWriter."""
    by_qdep: dict[int, list] = defaultdict(list)
    for r in rows:
        qd = r.get("QDep", 0)
        if qd <= max_qdep:
            by_qdep[qd].append(r)

    out = []
    for qd in sorted(by_qdep.keys()):
        pool = by_qdep[qd]
        sample = random.sample(pool, min(per_qdep, len(pool)))
        for i, r in enumerate(sample):
            out.append({
                "id": f"ProofWriter_qdep{qd}_{i}",
                "domain": "explicit_rule",
                "document_text": r["theory"],
                "query": r["question"],
                "gold_answer": str(r["answer"]),
                "hop_count": int(qd),
                "source_dataset": "ProofWriter",
                "split": "train",
            })
        logger.info(f"ProofWriter QDep {qd}: {len(sample)} instances")
    return out


@logger.catch(reraise=True)
def main() -> None:
    logger.info("Loading datasets...")

    clutrr = load_json(DS / "full_CLUTRR_v1_gen_train234_test2to10_test.json")
    babi = load_json(DS / "full_Muennighoff_babi_default_test.json")
    ruletaker = load_json(DS / "full_tasksource_ruletaker_default_train.json")
    proofwriter = load_json(DS / "full_tasksource_proofwriter_default_train.json")

    logger.info(f"Raw counts — CLUTRR: {len(clutrr)}, bAbI: {len(babi)}, RuleTaker: {len(ruletaker)}, ProofWriter: {len(proofwriter)}")

    rows: list[dict] = []
    rows.extend(process_clutrr(clutrr, target=20))
    rows.extend(process_babi(babi, tasks=[2, 3, 15], per_task=33))
    rows.extend(process_ruletaker(ruletaker, per_depth=25))
    rows.extend(process_proofwriter(proofwriter, per_qdep=25))

    logger.info(f"Total rows: {len(rows)}")
    by_src: dict[str, list] = defaultdict(list)
    for r in rows:
        by_src[r["source_dataset"]].append(r)
    for src, src_rows in sorted(by_src.items()):
        logger.info(f"  {src}: {len(src_rows)}")

    def row_to_example(r: dict) -> dict:
        prompt = f"Context: {r['document_text']}\n\nQuestion: {r['query']}"
        return {
            "input": prompt,
            "output": r["gold_answer"],
            "metadata_id": r["id"],
            "metadata_domain": r["domain"],
            "metadata_hop_count": r["hop_count"],
            "metadata_source_dataset": r["source_dataset"],
            "metadata_split": r["split"],
        }

    datasets = [
        {"dataset": src, "examples": [row_to_example(r) for r in src_rows]}
        for src, src_rows in sorted(by_src.items())
    ]
    doc = {
        "metadata": {
            "description": "Unified ATMS benchmark: CLUTRR kinship + bAbI tasks 2/3/15 + RuleTaker + ProofWriter",
            "total_examples": len(rows),
            "sources": {src: len(sr) for src, sr in sorted(by_src.items())},
        },
        "datasets": datasets,
    }

    out_path = WS / "data_out.json"
    out_path.write_text(json.dumps(doc, indent=2, ensure_ascii=False))
    logger.info(f"Wrote {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")

    # preview: first 10 examples from first dataset
    preview_doc = {
        "metadata": doc["metadata"],
        "datasets": [{"dataset": datasets[0]["dataset"], "examples": datasets[0]["examples"][:10]}],
    }
    (WS / "data_out_preview.json").write_text(json.dumps(preview_doc, indent=2, ensure_ascii=False))

    # mini: ~20% stratified by source
    mini_datasets = []
    for src, src_rows in sorted(by_src.items()):
        take = max(1, len(src_rows) // 5)
        sampled = random.sample(src_rows, take)
        mini_datasets.append({"dataset": src, "examples": [row_to_example(r) for r in sampled]})
    mini_doc = {"metadata": doc["metadata"], "datasets": mini_datasets}
    (WS / "data_out_mini.json").write_text(json.dumps(mini_doc, indent=2, ensure_ascii=False))
    mini_total = sum(len(d["examples"]) for d in mini_datasets)
    logger.info(f"Preview: 10 rows | Mini: {mini_total} rows")


if __name__ == "__main__":
    main()
