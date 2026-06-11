#!/usr/bin/env python3
"""Build full_data_out.json from downloaded datasets in temp/datasets/.

Datasets:
  babi_qa_task1       — bAbI task 1 (synthetic location-tracking, 200 stories)
  clutrr_v1           — CLUTRR kinship-chain reasoning, hops 3-8 (200 instances)
  hotpotqa            — HotpotQA Wikipedia multi-hop QA (200 instances)
  2wikimultihopqa     — 2WikiMultihopQA Wikipedia multi-hop QA (200 instances)

Output schema: exp_sel_data_out (datasets[].examples[].{input, output, metadata_*})
"""

import ast
import json
import random
import resource
import sys
from pathlib import Path

from loguru import logger

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add("logs/data.log", rotation="10 MB", level="DEBUG")

WORKSPACE = Path(__file__).parent
DATASETS_DIR = WORKSPACE / "temp" / "datasets"

random.seed(42)
resource.setrlimit(resource.RLIMIT_AS, (8 * 1024**3, 8 * 1024**3))

KINSHIP_LABELS = {
    0: "daughter", 1: "son", 2: "granddaughter", 3: "grandson",
    4: "mother", 5: "father", 6: "grandmother", 7: "grandfather",
    8: "sister", 9: "brother", 10: "niece", 11: "nephew",
    12: "aunt", 13: "uncle", 14: "wife", 15: "husband",
    16: "great-grandmother", 17: "great-grandfather",
}


def assign_fold(idx: int, total: int) -> str:
    r = idx / total
    if r < 0.70:
        return "train"
    elif r < 0.85:
        return "val"
    return "test"


def load_babi(path: Path, n: int = 200) -> list[dict]:
    """bAbI task 1: location-tracking multi-hop QA, l1_ambiguity=none."""
    logger.info(f"Loading bAbI from {path.name}")
    raw = json.loads(path.read_text())
    examples = []
    for row in raw[:n]:
        s = row["story"]
        types = s["type"]
        texts = s["text"]
        answers = s["answer"]
        supporting = s["supporting_ids"]

        context_lines: list[str] = []
        q_text = q_answer = None
        q_support: list[str] = []

        for t, txt, ans, sup in zip(types, texts, answers, supporting):
            if t == 0:
                context_lines.append(txt)
            elif q_text is None:
                q_text = txt
                q_answer = ans
                q_support = [
                    context_lines[int(x) - 1]
                    for x in sup
                    if int(x) - 1 < len(context_lines)
                ]

        if q_text is None or q_answer is None:
            continue

        input_doc = " ".join(context_lines)
        atomic_facts = []
        for sent in q_support:
            span = input_doc.find(sent)
            if span >= 0:
                atomic_facts.append({
                    "predicate": "location",
                    "args": [sent.split()[0], q_answer],
                    "span_start": span,
                    "span_end": span + len(sent),
                })

        idx = len(examples)
        examples.append({
            "input": f"Story: {input_doc}\n\nQuestion: {q_text}",
            "output": q_answer,
            "metadata_id": f"babi_qa1_{idx}",
            "metadata_fold": assign_fold(idx, n),
            "metadata_domain": "synthetic_reasoning",
            "metadata_hop_count": len(q_support),
            "metadata_l1_ambiguity_level": "none",
            "metadata_gold_atomic_facts": json.dumps(atomic_facts),
        })

    logger.info(f"  bAbI: {len(examples)} examples")
    return examples


def load_clutrr(path: Path, n: int = 200) -> list[dict]:
    """CLUTRR v1: kinship-chain reasoning hops 3-8, l1_ambiguity=none."""
    logger.info(f"Loading CLUTRR from {path.name}")
    raw = json.loads(path.read_text())

    filtered = [
        r for r in raw
        if "." in r.get("task_name", "")
        and 3 <= int(r["task_name"].split(".")[-1]) <= 8
    ]
    logger.info(f"  CLUTRR rows with hops 3-8: {len(filtered)}")
    selected = random.sample(filtered, min(n, len(filtered)))

    examples = []
    for i, row in enumerate(selected):
        story = row.get("story", row.get("clean_story", ""))
        hop_count = int(row["task_name"].split(".")[-1])

        target_text = row.get("target_text", "")
        if not target_text and isinstance(row.get("target"), int):
            target_text = KINSHIP_LABELS.get(row["target"], str(row["target"]))

        query_pair = row.get("query_edge", row.get("query", ""))
        if isinstance(query_pair, (list, tuple)) and len(query_pair) == 2:
            query_str = f"What is the relationship between {query_pair[0]} and {query_pair[1]}?"
        else:
            query_str = str(query_pair)

        # Parse story_edges and edge_types (stored as string repr)
        try:
            edges = ast.literal_eval(row.get("story_edges", "[]"))
            edge_types = ast.literal_eval(row.get("edge_types", "[]"))
        except Exception:
            edges, edge_types = [], []

        atomic_facts = []
        for edge, rel in zip(edges, edge_types):
            if isinstance(edge, (list, tuple)) and len(edge) == 2:
                atomic_facts.append({
                    "predicate": str(rel),
                    "args": [str(edge[0]), str(edge[1])],
                    "span_start": 0,
                    "span_end": 0,
                })

        examples.append({
            "input": f"Story: {story}\n\nQuestion: {query_str}",
            "output": target_text,
            "metadata_id": f"clutrr_v1_{i}",
            "metadata_fold": assign_fold(i, len(selected)),
            "metadata_domain": "kinship_reasoning",
            "metadata_hop_count": hop_count,
            "metadata_l1_ambiguity_level": "none",
            "metadata_gold_atomic_facts": json.dumps(atomic_facts),
        })

    logger.info(f"  CLUTRR: {len(examples)} examples")
    return examples


def load_hotpotqa(path: Path, n: int = 200) -> list[dict]:
    """HotpotQA: Wikipedia distractor multi-hop QA, l1_ambiguity=low."""
    logger.info(f"Loading HotpotQA from {path.name}")
    raw = json.loads(path.read_text())

    distractor = [r for r in raw if r.get("type") == "distractor"]
    pool = distractor if len(distractor) >= n else raw
    selected = random.sample(pool, min(n, len(pool)))

    examples = []
    for i, row in enumerate(selected):
        ctx = row["context"]
        titles = ctx["title"]
        sentences = ctx["sentences"]

        sf = row.get("supporting_facts", {})
        sf_titles = set(sf.get("title", []))
        sf_titles_list = sf.get("title", [])
        sf_sent_ids = sf.get("sent_id", [])

        passages = []
        for title, sents in zip(titles, sentences):
            marker = "[SUPPORTING] " if title in sf_titles else ""
            passages.append(f"{marker}{title}: {' '.join(sents)}")
        input_doc = "\n\n".join(passages)

        atomic_facts = []
        for title, sent_id in zip(sf_titles_list, sf_sent_ids):
            if title in titles:
                tidx = titles.index(title)
                sents = sentences[tidx]
                if sent_id < len(sents):
                    sent_text = sents[sent_id]
                    span = input_doc.find(sent_text)
                    atomic_facts.append({
                        "predicate": "supports",
                        "args": [title, row["answer"]],
                        "span_start": max(span, 0),
                        "span_end": max(span + len(sent_text), 0),
                    })

        examples.append({
            "input": f"Context:\n{input_doc}\n\nQuestion: {row['question']}",
            "output": row["answer"],
            "metadata_id": f"hotpotqa_{row['id']}",
            "metadata_fold": assign_fold(i, len(selected)),
            "metadata_domain": "wikipedia_multihop",
            "metadata_hop_count": 2,
            "metadata_l1_ambiguity_level": "low",
            "metadata_gold_atomic_facts": json.dumps(atomic_facts),
        })

    logger.info(f"  HotpotQA: {len(examples)} examples")
    return examples


def load_2wiki(path: Path, n: int = 200) -> list[dict]:
    """2WikiMultihopQA: structured+unstructured multi-hop, l1_ambiguity=medium/high."""
    logger.info(f"Loading 2WikiMultihopQA from {path.name}")
    raw = json.loads(path.read_text())
    selected = random.sample(raw, min(n, len(raw)))

    examples = []
    for i, row in enumerate(selected):
        ctx = row.get("context", {})
        titles = ctx.get("title", [])
        sentences = ctx.get("sentences", [])

        sf = row.get("supporting_facts", {})
        sf_titles = set(sf.get("title", []))

        passages = []
        for title, sents in zip(titles, sentences):
            marker = "[SUPPORTING] " if title in sf_titles else ""
            passages.append(f"{marker}{title}: {' '.join(sents)}")
        input_doc = "\n\n".join(passages)

        atomic_facts = []
        for ev in row.get("evidences", []):
            if isinstance(ev, (list, tuple)) and len(ev) == 3:
                subj, pred, obj = ev
                span = input_doc.find(str(subj))
                atomic_facts.append({
                    "predicate": str(pred),
                    "args": [str(subj), str(obj)],
                    "span_start": max(span, 0),
                    "span_end": max(span + len(str(subj)), 0),
                })

        hop_type = row.get("type", "bridge")
        l1_level = "high" if hop_type == "comparison" else "medium"
        hop_count = 2 if hop_type == "bridge" else 3

        examples.append({
            "input": f"Context:\n{input_doc}\n\nQuestion: {row['question']}",
            "output": row["answer"],
            "metadata_id": f"2wiki_{row['id']}",
            "metadata_fold": assign_fold(i, len(selected)),
            "metadata_domain": "wikipedia_multihop",
            "metadata_hop_count": hop_count,
            "metadata_l1_ambiguity_level": l1_level,
            "metadata_gold_atomic_facts": json.dumps(atomic_facts),
        })

    logger.info(f"  2WikiMultihopQA: {len(examples)} examples")
    return examples


@logger.catch(reraise=True)
def main() -> None:
    logger.info("Starting data.py")
    (WORKSPACE / "logs").mkdir(exist_ok=True)

    # Best 3 datasets selected for L1 ambiguity research:
    # clutrr_v1 (none), hotpotqa (low), 2wikimultihopqa (medium/high)
    # bAbI task1 dropped: only 1-hop location tracking, zero ambiguity, weak signal
    clutrr = load_clutrr(DATASETS_DIR / "full_kendrivp_CLUTRR_v1_extracted_default_train.json", n=200)
    hotpot = load_hotpotqa(DATASETS_DIR / "full_hotpotqa_hotpot_qa_fullwiki_validation.json", n=200)
    wiki2 = load_2wiki(DATASETS_DIR / "full_framolfese_2WikiMultihopQA_default_validation.json", n=200)

    total = len(clutrr) + len(hotpot) + len(wiki2)
    logger.info(f"Total examples: {total}")
    assert total >= 550, f"Only {total} examples, need ≥550"

    output = {
        "metadata": {
            "description": "L1 grounding ambiguity dataset for multi-hop reasoning",
            "total_examples": total,
            "sources": ["clutrr_v1", "hotpotqa", "2wikimultihopqa"],
        },
        "datasets": [
            {"dataset": "clutrr_v1", "examples": clutrr},
            {"dataset": "hotpotqa", "examples": hotpot},
            {"dataset": "2wikimultihopqa", "examples": wiki2},
        ],
    }

    out_path = WORKSPACE / "full_data_out.json"
    out_path.write_text(json.dumps(output, indent=2))
    logger.info(f"Saved → {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")

    stats = {
        "total_examples": total,
        "by_dataset": {
            "clutrr_v1": len(clutrr),
            "hotpotqa": len(hotpot),
            "2wikimultihopqa": len(wiki2),
        },
        "by_fold": {
            fold: sum(
                sum(1 for ex in ds["examples"] if ex["metadata_fold"] == fold)
                for ds in output["datasets"]
            )
            for fold in ["train", "val", "test"]
        },
        "by_domain": {},
        "by_l1_ambiguity_level": {},
    }
    all_examples = [ex for ds in output["datasets"] for ex in ds["examples"]]
    for ex in all_examples:
        d = ex["metadata_domain"]
        stats["by_domain"][d] = stats["by_domain"].get(d, 0) + 1
        l = ex["metadata_l1_ambiguity_level"]
        stats["by_l1_ambiguity_level"][l] = stats["by_l1_ambiguity_level"].get(l, 0) + 1

    (WORKSPACE / "dataset_stats.json").write_text(json.dumps(stats, indent=2))
    logger.info(f"Stats: {stats['by_dataset']}  folds: {stats['by_fold']}")


if __name__ == "__main__":
    main()
