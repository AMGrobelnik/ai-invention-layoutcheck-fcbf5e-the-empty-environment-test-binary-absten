#!/usr/bin/env python3
"""Transform downloaded datasets into unified exp_sel_data_out schema."""

import ast
import json
import math
import random
import resource
import sys
from pathlib import Path

from loguru import logger

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add("logs/collect_data.log", rotation="10 MB", level="DEBUG")

WORKSPACE = Path(__file__).parent
DATASETS_DIR = WORKSPACE / "temp" / "datasets"
OUTPUT_DIR = WORKSPACE

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


def process_babi(path: Path, n: int = 200) -> list[dict]:
    logger.info(f"Processing bAbI task1 from {path.name}")
    raw = json.loads(path.read_text())
    examples = []
    for row in raw[:n]:
        s = row["story"]
        ids = s["id"]
        types = s["type"]
        texts = s["text"]
        answers = s["answer"]
        supporting = s["supporting_ids"]

        # Build context: all statement-type (type==0) sentences
        # Questions are type==1; find first question
        context_lines = []
        q_text = None
        q_answer = None
        q_support = []
        for i, (t, txt, ans, sup) in enumerate(zip(types, texts, answers, supporting)):
            if t == 0:
                context_lines.append(txt)
            else:
                if q_text is None:
                    q_text = txt
                    q_answer = ans
                    q_support = [context_lines[int(x) - 1] for x in sup if int(x) - 1 < len(context_lines)]

        if q_text is None or q_answer is None:
            continue

        input_doc = " ".join(context_lines)
        atomic_facts = [
            {"predicate": "location", "args": [sent.split()[0], q_answer],
             "span_start": input_doc.find(sent), "span_end": input_doc.find(sent) + len(sent)}
            for sent in q_support if sent and input_doc.find(sent) >= 0
        ]

        examples.append({
            "input": f"Story: {input_doc}\n\nQuestion: {q_text}",
            "output": q_answer,
            "metadata_id": f"babi_qa1_{len(examples)}",
            "metadata_fold": assign_fold(len(examples), n),
            "metadata_domain": "synthetic_reasoning",
            "metadata_hop_count": 2,
            "metadata_l1_ambiguity_level": "none",
            "metadata_gold_atomic_facts": json.dumps(atomic_facts),
        })
    logger.info(f"  → {len(examples)} bAbI examples")
    return examples


def process_clutrr(path: Path, n: int = 200) -> list[dict]:
    logger.info(f"Processing CLUTRR v1 from {path.name}")
    raw = json.loads(path.read_text())

    # Filter hops 3-8
    filtered = [
        r for r in raw
        if "." in r.get("task_name", "") and 3 <= int(r["task_name"].split(".")[-1]) <= 8
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

        # Atomic facts from story_edges
        atomic_facts = []
        edges_raw = row.get("story_edges", [])
        edge_types_raw = row.get("edge_types", [])
        try:
            edges = ast.literal_eval(edges_raw) if isinstance(edges_raw, str) else edges_raw
            edge_types = ast.literal_eval(edge_types_raw) if isinstance(edge_types_raw, str) else edge_types_raw
        except Exception:
            edges, edge_types = [], []
        if edges and edge_types:
            for edge, rel in zip(edges, edge_types):
                if not (isinstance(edge, (list, tuple)) and len(edge) == 2):
                    continue
                src, tgt = str(edge[0]), str(edge[1])
                atomic_facts.append({
                    "predicate": str(rel),
                    "args": [src, tgt],
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
    logger.info(f"  → {len(examples)} CLUTRR examples")
    return examples


def process_hotpotqa(path: Path, n: int = 200) -> list[dict]:
    logger.info(f"Processing HotpotQA from {path.name}")
    raw = json.loads(path.read_text())

    # Prefer distractor type for harder multi-hop
    distractor = [r for r in raw if r.get("type") == "distractor"]
    pool = distractor if len(distractor) >= n else raw
    selected = random.sample(pool, min(n, len(pool)))

    examples = []
    for i, row in enumerate(selected):
        ctx = row["context"]
        titles = ctx["title"]
        sentences = ctx["sentences"]

        # Identify supporting passages
        sf = row.get("supporting_facts", {})
        sf_titles = set(sf.get("title", []))

        # Build document from all context passages
        passages = []
        for title, sents in zip(titles, sentences):
            marker = "[SUPPORTING] " if title in sf_titles else ""
            passages.append(f"{marker}{title}: {' '.join(sents)}")
        input_doc = "\n\n".join(passages)

        # Atomic facts from supporting sentences
        sf_sent_ids = sf.get("sent_id", [])
        sf_titles_list = sf.get("title", [])
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
    logger.info(f"  → {len(examples)} HotpotQA examples")
    return examples


def process_2wiki(path: Path, n: int = 200) -> list[dict]:
    logger.info(f"Processing 2WikiMultihopQA from {path.name}")
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

        # Atomic facts from evidences (Wikidata triples)
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

        examples.append({
            "input": f"Context:\n{input_doc}\n\nQuestion: {row['question']}",
            "output": row["answer"],
            "metadata_id": f"2wiki_{row['id']}",
            "metadata_fold": assign_fold(i, len(selected)),
            "metadata_domain": "wikipedia_multihop",
            "metadata_hop_count": 2 if hop_type == "bridge" else 3,
            "metadata_l1_ambiguity_level": l1_level,
            "metadata_gold_atomic_facts": json.dumps(atomic_facts),
        })
    logger.info(f"  → {len(examples)} 2WikiMultihopQA examples")
    return examples


@logger.catch(reraise=True)
def main() -> None:
    logger.info("Starting dataset collection")
    (WORKSPACE / "logs").mkdir(exist_ok=True)

    babi = process_babi(DATASETS_DIR / "full_babi_qa_en10k_qa1.json", n=200)
    clutrr = process_clutrr(DATASETS_DIR / "full_kendrivp_CLUTRR_v1_extracted_default_train.json", n=200)
    hotpot = process_hotpotqa(DATASETS_DIR / "full_hotpotqa_hotpot_qa_fullwiki_validation.json", n=200)
    wiki2 = process_2wiki(DATASETS_DIR / "full_framolfese_2WikiMultihopQA_default_validation.json", n=200)

    total = len(babi) + len(clutrr) + len(hotpot) + len(wiki2)
    logger.info(f"Total examples: {total} (target ≥550)")
    assert total >= 550, f"Only {total} examples, need ≥550"

    output = {
        "metadata": {
            "description": "L1 grounding ambiguity dataset for multi-hop reasoning",
            "total_examples": total,
            "sources": ["babi_qa_task1", "clutrr_v1", "hotpotqa", "2wikimultihopqa"],
        },
        "datasets": [
            {"dataset": "babi_qa_task1", "examples": babi},
            {"dataset": "clutrr_v1", "examples": clutrr},
            {"dataset": "hotpotqa", "examples": hotpot},
            {"dataset": "2wikimultihopqa", "examples": wiki2},
        ],
    }

    out_path = OUTPUT_DIR / "full_data_out.json"
    out_path.write_text(json.dumps(output, indent=2))
    logger.info(f"Saved → {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")

    # Stats
    stats = {
        "total_examples": total,
        "by_dataset": {
            "babi_qa_task1": len(babi),
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
        stats["by_domain"][ex["metadata_domain"]] = stats["by_domain"].get(ex["metadata_domain"], 0) + 1
        stats["by_l1_ambiguity_level"][ex["metadata_l1_ambiguity_level"]] = (
            stats["by_l1_ambiguity_level"].get(ex["metadata_l1_ambiguity_level"], 0) + 1
        )

    stats_path = OUTPUT_DIR / "dataset_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2))
    logger.info(f"Stats → {stats_path}")
    logger.info(f"  by_dataset: {stats['by_dataset']}")
    logger.info(f"  by_fold:    {stats['by_fold']}")
    logger.info(f"  by_domain:  {stats['by_domain']}")
    logger.info(f"  by_l1:      {stats['by_l1_ambiguity_level']}")


if __name__ == "__main__":
    main()
