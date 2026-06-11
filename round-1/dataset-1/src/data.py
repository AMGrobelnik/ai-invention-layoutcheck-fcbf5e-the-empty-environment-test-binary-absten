#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["loguru"]
# ///
"""Build full_data_out.json: unified ATMS benchmark from CLUTRR + RuleTaker + ProofWriter."""

import ast
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from loguru import logger

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")

WS = Path(__file__).parent
DS = WS / "temp" / "datasets"
random.seed(42)


def load_json(path: Path) -> list:
    return json.loads(path.read_text())


@logger.catch(reraise=True)
def process_clutrr(rows: list, target: int = 20) -> list:
    by_hop: dict[int, list] = defaultdict(list)
    for r in rows:
        hop = len(ast.literal_eval(r["edge_types"]))
        by_hop[hop].append(r)
    out = []
    for hop in sorted(by_hop):
        sample = random.sample(by_hop[hop], min(target, len(by_hop[hop])))
        for r in sample:
            names = ast.literal_eval(r["query"])
            out.append({
                "id": f"CLUTRR_test_{r['id']}",
                "domain": "kinship",
                "document_text": r["story"],
                "query": f"What is the relationship between {names[0]} and {names[1]}?",
                "gold_answer": r["target_text"],
                "hop_count": hop,
                "source_dataset": "CLUTRR",
                "split": "test",
            })
    logger.info(f"CLUTRR: {len(out)} instances, hops {sorted(by_hop.keys())}")
    return out


@logger.catch(reraise=True)
def process_ruletaker(
    rows: list,
    depths: list[str] = ["depth-0", "depth-1", "depth-2", "depth-3"],
    per_depth: int = 25,
) -> list:
    depth_to_hop = {"depth-0": 0, "depth-1": 1, "depth-2": 2, "depth-3": 3}
    by_depth: dict[str, list] = defaultdict(list)
    for r in rows:
        if r["config"] in depths:
            by_depth[r["config"]].append(r)
    out = []
    for d in depths:
        sample = random.sample(by_depth[d], min(per_depth, len(by_depth[d])))
        for i, r in enumerate(sample):
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
    by_qdep: dict[int, list] = defaultdict(list)
    for r in rows:
        qd = r.get("QDep", 0)
        if qd <= max_qdep:
            by_qdep[qd].append(r)
    out = []
    for qd in sorted(by_qdep.keys()):
        sample = random.sample(by_qdep[qd], min(per_qdep, len(by_qdep[qd])))
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


def row_to_example(r: dict) -> dict:
    return {
        "input": f"Context: {r['document_text']}\n\nQuestion: {r['query']}",
        "output": r["gold_answer"],
        "metadata_id": r["id"],
        "metadata_domain": r["domain"],
        "metadata_hop_count": r["hop_count"],
        "metadata_source_dataset": r["source_dataset"],
        "metadata_split": r["split"],
    }


@logger.catch(reraise=True)
def main() -> None:
    logger.info("Loading raw datasets...")
    clutrr = load_json(DS / "full_CLUTRR_v1_gen_train234_test2to10_test.json")
    ruletaker = load_json(DS / "full_tasksource_ruletaker_default_train.json")
    proofwriter = load_json(DS / "full_tasksource_proofwriter_default_train.json")
    logger.info(f"Raw counts — CLUTRR:{len(clutrr)} RuleTaker:{len(ruletaker)} ProofWriter:{len(proofwriter)}")

    rows: list[dict] = []
    rows.extend(process_clutrr(clutrr, target=20))
    rows.extend(process_ruletaker(ruletaker, per_depth=25))
    rows.extend(process_proofwriter(proofwriter, per_qdep=25))

    by_src: dict[str, list] = defaultdict(list)
    for r in rows:
        by_src[r["source_dataset"]].append(r)

    logger.info(f"Total rows: {len(rows)}")
    for src, src_rows in sorted(by_src.items()):
        logger.info(f"  {src}: {len(src_rows)}")

    doc = {
        "metadata": {
            "description": "Unified ATMS benchmark: CLUTRR kinship + RuleTaker + ProofWriter explicit-rule",
            "total_examples": len(rows),
            "sources": {src: len(sr) for src, sr in sorted(by_src.items())},
        },
        "datasets": [
            {"dataset": src, "examples": [row_to_example(r) for r in src_rows]}
            for src, src_rows in sorted(by_src.items())
        ],
    }

    out_path = WS / "full_data_out.json"
    out_path.write_text(json.dumps(doc, indent=2, ensure_ascii=False))
    logger.info(f"Wrote {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
