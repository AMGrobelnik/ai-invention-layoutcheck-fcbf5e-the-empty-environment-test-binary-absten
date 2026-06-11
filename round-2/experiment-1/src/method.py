#!/usr/bin/env python3
"""
Bounded ATMS Iter-2: Matched-Coverage Baselines, 70B Ablation, Stratification Fix.
Extends iter-1 with: matched-coverage baselines, 70B model ablation,
NaN guards, ATMS-vs-ProbLog ground-truth annotation, assumption-load stratification.
Loads 380 instances from the dependency dataset artifact.
"""

from __future__ import annotations

import asyncio
import gc
import json
import math
import os
import random
import re
import resource
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import aiohttp
import numpy as np
from loguru import logger
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from tenacity import retry, stop_after_attempt, wait_exponential

# ─── Workspace ────────────────────────────────────────────────────────────────
WORKSPACE = Path(__file__).parent
LOGS_DIR = WORKSPACE / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# ─── Logging ──────────────────────────────────────────────────────────────────
logger.remove()
GREEN, CYAN, END = "\033[92m", "\033[96m", "\033[0m"
logger.add(
    sys.stdout,
    level="INFO",
    format=f"{GREEN}{{time:HH:mm:ss}}{END}|{{level:<7}}|{CYAN}{{function}}{END}| {{message}}",
)
logger.add(str(LOGS_DIR / "run.log"), rotation="30 MB", level="DEBUG")

# Memory limit: 16 GB virtual (all LLM-based, lightweight)
try:
    resource.setrlimit(resource.RLIMIT_AS, (16 * 1024**3, 16 * 1024**3))
except ValueError:
    pass

# ─── Config ───────────────────────────────────────────────────────────────────
OPENROUTER_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE = "https://openrouter.ai/api/v1"
CHEAP_MODEL = "meta-llama/llama-3.1-8b-instruct"
LARGE_MODEL = "meta-llama/llama-3.1-70b-instruct"
FALLBACK_MODELS = ["google/gemini-flash-1.5-8b", "mistralai/mistral-7b-instruct"]
COST_CAP = 8.0
ASSUMPTION_THRESHOLD = 2
DEPTH_CAP = 3
BEAM_WIDTH = 20
LINC_K = 3
TARGET_COVERAGE = 0.229
RANDOM_ORACLE_SEEDS = 100

DATASET_PATH = Path(
    "/home/adrian/projects/ai-inventor/aii_data/users/admin/runs/"
    "run_r4ZXzXG-rGek/3_invention_loop/iter_1/gen_art/gen_art_dataset_1/full_data_out.json"
)

# ─── Cost Tracking ────────────────────────────────────────────────────────────
_cumulative_cost: float = 0.0

PRICE_PER_1M = {
    "meta-llama/llama-3.1-8b-instruct": {"in": 0.06, "out": 0.06},
    "meta-llama/llama-3.1-70b-instruct": {"in": 0.27, "out": 0.27},
    "google/gemini-flash-1.5-8b": {"in": 0.0375, "out": 0.15},
    "mistralai/mistral-7b-instruct": {"in": 0.055, "out": 0.055},
    "anthropic/claude-haiku-4-5": {"in": 0.80, "out": 4.0},
}

_current_model: str = CHEAP_MODEL
_model_failed: set[str] = set()


def estimate_cost(prompt_tokens: int, completion_tokens: int, model: str) -> float:
    prices = PRICE_PER_1M.get(model, {"in": 0.5, "out": 1.5})
    return (prompt_tokens * prices["in"] + completion_tokens * prices["out"]) / 1_000_000


# ─── HTTP Session ─────────────────────────────────────────────────────────────
_session: aiohttp.ClientSession | None = None
_semaphore: asyncio.Semaphore | None = None


async def get_session() -> aiohttp.ClientSession:
    global _session
    if _session is None or _session.closed:
        timeout = aiohttp.ClientTimeout(total=120)
        _session = aiohttp.ClientSession(timeout=timeout)
    return _session


async def get_semaphore() -> asyncio.Semaphore:
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(8)
    return _semaphore


async def close_session() -> None:
    global _session
    if _session and not _session.closed:
        await _session.close()


# ─── LLM Caller ──────────────────────────────────────────────────────────────
async def llm_call(
    prompt: str,
    max_tokens: int = 600,
    retries: int = 3,
    model_override: str | None = None,
) -> tuple[str, float]:
    """Call OpenRouter LLM. Returns (response_text, cost_usd)."""
    global _cumulative_cost, _current_model

    if _cumulative_cost >= COST_CAP:
        raise RuntimeError(f"Cost cap ${COST_CAP} reached: ${_cumulative_cost:.4f}")

    session = await get_session()
    sem = await get_semaphore()

    use_model = model_override or _current_model
    models_to_try = [use_model] + [m for m in FALLBACK_MODELS if m not in _model_failed and m != use_model]

    async with sem:
        for model in models_to_try:
            for attempt in range(retries):
                try:
                    payload = {
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": max_tokens,
                        "temperature": 0.1,
                    }
                    headers = {
                        "Authorization": f"Bearer {OPENROUTER_KEY}",
                        "Content-Type": "application/json",
                    }
                    async with session.post(
                        f"{OPENROUTER_BASE}/chat/completions",
                        json=payload,
                        headers=headers,
                    ) as resp:
                        if resp.status == 429:
                            await asyncio.sleep(5 * (attempt + 1))
                            continue
                        if resp.status >= 400:
                            text = await resp.text()
                            logger.warning(f"API error {resp.status} for {model}: {text[:200]}")
                            if resp.status in (400, 404):
                                _model_failed.add(model)
                            await asyncio.sleep(2)
                            break
                        data = await resp.json()
                        content = data["choices"][0]["message"]["content"] or ""
                        usage = data.get("usage", {})
                        cost = estimate_cost(
                            usage.get("prompt_tokens", 0),
                            usage.get("completion_tokens", 0),
                            model,
                        )
                        _cumulative_cost += cost
                        logger.debug(
                            f"LLM [{model}] cost=${cost:.5f} total=${_cumulative_cost:.4f}"
                        )
                        return content, cost
                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    logger.warning(f"LLM call attempt {attempt+1} failed: {e}")
                    await asyncio.sleep(2 * (attempt + 1))
            _model_failed.add(model)

    raise RuntimeError("All LLM models failed")


def extract_json(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    try:
        import json5
        return json5.loads(text)
    except Exception:
        pass
    for pattern in [r'\[.*?\]', r'\{.*?\}']:
        m = re.search(pattern, text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# BOUNDED ATMS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Assumption:
    id: str
    text: str
    weight: float


@dataclass
class Node:
    id: str
    label: set

    def __post_init__(self):
        if not isinstance(self.label, set):
            self.label = set(self.label)


class BoundedATMS:
    def __init__(self, depth_cap: int = 3, beam_width: int = 20):
        self.depth_cap = depth_cap
        self.beam_width = beam_width
        self.assumptions: dict[str, Assumption] = {}
        self.nodes: dict[str, Node] = {}
        self.nogoods: list[frozenset] = []
        self._rules: list[tuple] = []

    def _add_env_to_label(self, node_id: str, env: frozenset) -> bool:
        node = self.nodes[node_id]
        for existing in node.label:
            if existing <= env:
                return False
        if len(env) > self.depth_cap:
            return False
        for ng in self.nogoods:
            if ng <= env:
                return False
        to_remove = {e for e in node.label if env < e}
        node.label -= to_remove
        node.label.add(env)
        if len(node.label) > self.beam_width:
            sorted_envs = sorted(node.label, key=lambda e: (len(e), sorted(e)))
            node.label = set(sorted_envs[: self.beam_width])
        return True

    def _get_or_create(self, node_id: str) -> Node:
        if node_id not in self.nodes:
            self.nodes[node_id] = Node(id=node_id, label=set())
        return self.nodes[node_id]

    def add_fact(self, node_id: str, env: frozenset | None = None) -> None:
        if env is None:
            env = frozenset()
        self._get_or_create(node_id)
        self._add_env_to_label(node_id, env)

    def add_assumption(self, assumption: Assumption) -> str:
        self.assumptions[assumption.id] = assumption
        self._get_or_create(assumption.id)
        self._add_env_to_label(assumption.id, frozenset([assumption.id]))
        return assumption.id

    def add_rule(self, antecedents: list[str], consequent: str, via_assumption: str | None = None) -> None:
        rule_env_extra = frozenset([via_assumption]) if via_assumption else frozenset()
        self._get_or_create(consequent)
        for ant in antecedents:
            self._get_or_create(ant)
        self._rules.append((antecedents, consequent, rule_env_extra))
        self._fire_rule(antecedents, consequent, rule_env_extra)

    def _fire_rule(self, antecedents: list[str], consequent: str, rule_env_extra: frozenset) -> bool:
        changed = False
        ant_labels = []
        for ant in antecedents:
            node = self.nodes.get(ant)
            if node is None or not node.label:
                return False
            ant_labels.append(list(node.label))

        def product_envs(labels: list[list[frozenset]], limit: int = 200):
            if not labels:
                yield frozenset()
                return
            first, rest = labels[0], labels[1:]
            count = 0
            for env in first:
                for rest_env in product_envs(rest, limit):
                    yield env | rest_env
                    count += 1
                    if count >= limit:
                        return

        for combined in product_envs(ant_labels):
            final_env = combined | rule_env_extra
            if self._add_env_to_label(consequent, final_env):
                changed = True
        return changed

    def add_nogood(self, assumption_ids: frozenset) -> None:
        self.nogoods.append(assumption_ids)
        for node in self.nodes.values():
            to_remove = {e for e in node.label if assumption_ids <= e}
            node.label -= to_remove

    def merge_nodes(self, node_id1: str, node_id2: str, via_assumption: str) -> None:
        n1 = self.nodes.get(node_id1)
        n2 = self.nodes.get(node_id2)
        assumption_env = frozenset([via_assumption])
        if n1:
            for env in list(n1.label):
                self._add_env_to_label(node_id2, env | assumption_env)
        if n2:
            for env in list(n2.label):
                self._add_env_to_label(node_id1, env | assumption_env)

    def propagate(self) -> None:
        for _ in range(10):
            changed = False
            for antecedents, consequent, rule_env_extra in self._rules:
                if self._fire_rule(antecedents, consequent, rule_env_extra):
                    changed = True
            if not changed:
                break

    def assumption_load(self, node_id: str) -> float:
        node = self.nodes.get(node_id)
        if node is None or not node.label:
            return float("inf")
        if frozenset() in node.label:
            return 0.0
        return float(min(len(e) for e in node.label))

    def is_empty_env_derivable(self, node_id: str) -> bool:
        node = self.nodes.get(node_id)
        if node is None:
            return False
        return frozenset() in node.label


# ═══════════════════════════════════════════════════════════════════════════════
# KINSHIP / LOCATION RULES
# ═══════════════════════════════════════════════════════════════════════════════

_PARENT_PREDS = re.compile(
    r'^(parent|father|mother|dad|mom|is_parent|is_father|is_mother|parent_of)\((.+?),(.+?)\)$',
    re.IGNORECASE,
)
_SIBLING_PREDS = re.compile(
    r'^(sibling|brother|sister|is_sibling|sibling_of)\((.+?),(.+?)\)$',
    re.IGNORECASE,
)
_TRAVELED_PREDS = re.compile(
    r'^(traveled_to|went_to|moved_to|went|traveled|at|is_at|location)\((.+?),(.+?)\)$',
    re.IGNORECASE,
)
_PICKUP_PREDS = re.compile(r'^(picked_up|picked|grabbed|has)\((.+?),(.+?)\)$', re.IGNORECASE)
_DROP_PREDS = re.compile(r'^(dropped|put_down|left)\((.+?),(.+?)\)$', re.IGNORECASE)


def _apply_kinship_rules(atms: BoundedATMS) -> None:
    node_ids = list(atms.nodes.keys())
    parents: list[tuple[str, str, str]] = []
    siblings: list[tuple[str, str, str]] = []
    for nid in node_ids:
        m = _PARENT_PREDS.match(nid)
        if m:
            parents.append((m.group(2).strip(), m.group(3).strip(), nid))
            if not nid.startswith("parent("):
                can_nid = f"parent({m.group(2).strip()},{m.group(3).strip()})"
                atms.add_fact(can_nid, env=frozenset())
                parents.append((m.group(2).strip(), m.group(3).strip(), can_nid))
        m = _SIBLING_PREDS.match(nid)
        if m:
            siblings.append((m.group(2).strip(), m.group(3).strip(), nid))

    seen_parent_pairs = set()
    for x, y1, nid1 in parents:
        for y2, z, nid2 in parents:
            if y1 == y2 and (x, z) not in seen_parent_pairs:
                seen_parent_pairs.add((x, z))
                gp_node = f"grandparent({x},{z})"
                atms._get_or_create(gp_node)
                atms.add_rule([nid1, nid2], gp_node)
                for y3, w, nid3 in parents:
                    if y3 == z:
                        ggp_node = f"great_grandparent({x},{w})"
                        atms._get_or_create(ggp_node)
                        atms.add_rule([nid1, nid2, nid3], ggp_node)

    for x, y, nid in siblings:
        sym_node = f"sibling({y},{x})"
        atms._get_or_create(sym_node)
        atms.add_rule([nid], sym_node)

    for px, py, pnid in parents:
        for sx, sy, snid in siblings:
            if py == sx:
                rel_node = f"aunt_or_uncle({px},{sy})"
                atms._get_or_create(rel_node)
                atms.add_rule([pnid, snid], rel_node)


def _apply_location_rules(atms: BoundedATMS) -> None:
    node_ids = list(atms.nodes.keys())
    ats: list[tuple[str, str, str]] = []

    for nid in node_ids:
        m = _TRAVELED_PREDS.match(nid)
        if m:
            p, loc = m.group(2).strip(), m.group(3).strip()
            at_node = f"at({p},{loc})"
            atms.add_fact(at_node, env=frozenset())
            ats.append((p, loc, at_node))
        m = _PICKUP_PREDS.match(nid)
        if m:
            person, obj = m.group(2).strip(), m.group(3).strip()
            has_node = f"has({person},{obj})"
            atms.add_fact(has_node, env=frozenset())

    has_facts = [
        (m.group(1), m.group(2), nid)
        for nid in node_ids
        if (m := re.match(r'has\((.+?),(.+?)\)', nid))
    ]
    for p1, loc, at_nid in ats:
        for p2, obj, has_nid in has_facts:
            if p1 == p2:
                obj_at_node = f"at({obj},{loc})"
                atms._get_or_create(obj_at_node)
                atms.add_rule([at_nid, has_nid], obj_at_node)

    for p1, loc1, nid1 in ats:
        for p2, loc2, nid2 in ats:
            if loc1 == loc2 and p1 != p2:
                sl_node = f"same_location({p1},{p2})"
                atms._get_or_create(sl_node)
                atms.add_rule([nid1, nid2], sl_node)


# ═══════════════════════════════════════════════════════════════════════════════
# L1 GROUNDING
# ═══════════════════════════════════════════════════════════════════════════════

L1_PROMPT = """\
You are a precise fact extractor. Extract ONLY atomic facts explicitly stated in the document below.
No inference, no world knowledge, no assumptions.

Each fact must be a simple predicate-argument structure. Return a JSON array of objects with:
- "predicate": the relation using CANONICAL names below
- "arg1": first argument (entity name, lowercased)
- "arg2": second argument (entity name or value, lowercased), or "" if unary
- "span": the exact substring from the document that states this fact

CANONICAL predicates to use:
- Family relations: "parent" (for mother/father/parent-of), "sibling" (for brother/sister), "married" (for spouse)
- Location: "traveled_to", "at", "picked_up", "dropped", "has"
- Other: use snake_case short forms (e.g. "owns", "works_at", "signed", "filed_suit")

Return ONLY the JSON array. If no facts found, return [].

Document:
{document}"""


async def extract_l1_atoms(document: str) -> tuple[list[dict], float]:
    prompt = L1_PROMPT.format(document=document[:3000])
    resp, cost = await llm_call(prompt, max_tokens=800)
    atoms = extract_json(resp)
    if not isinstance(atoms, list):
        logger.warning(f"L1 parse failed: {resp[:100]}")
        atoms = []

    doc_lower = document.lower()
    for atom in atoms:
        if not isinstance(atom, dict):
            continue
        span = atom.get("span", "")
        atom["grounding_verified"] = span.lower() in doc_lower if span else False
        for k in ("predicate", "arg1", "arg2"):
            atom.setdefault(k, "")
        atom["node_id"] = (
            f"{atom['predicate']}({atom['arg1']},{atom['arg2']})"
            if atom.get("arg2")
            else f"{atom['predicate']}({atom['arg1']})"
        )

    atoms = [a for a in atoms if isinstance(a, dict) and a.get("predicate")]
    return atoms, cost


# ═══════════════════════════════════════════════════════════════════════════════
# L2 ASSUMPTION PROPOSAL
# ═══════════════════════════════════════════════════════════════════════════════

L2_PROMPT = """\
Given document atoms (explicit facts) and a query, identify what ADDITIONAL assumptions are needed to answer the query.

Document atoms (JSON):
{atoms_json}

Query: {query}

Propose:
1. "unifications": pairs of atoms that MIGHT refer to the same entity/relation but use different names. Only include if genuinely uncertain.
2. "bridge_rules": common-sense inference rules (not stated in document) needed to connect atoms to answer the query.

Return ONLY valid JSON:
{{"unifications": [{{"atom1": "node_id_1", "atom2": "node_id_2", "weight": 0.0-1.0, "rationale": "..."}}],
  "bridge_rules": [{{"antecedent": ["node_id"], "consequent": "derived_node_id", "weight": 0.0-1.0, "rationale": "..."}}]}}

Weight = confidence this assumption is correct (0=uncertain, 1=certain).
If no assumptions needed (query is directly answerable), return {{"unifications": [], "bridge_rules": []}}.
Keep total assumptions under 15. Return ONLY JSON."""


async def propose_l2_assumptions(
    l1_atoms: list[dict],
    document: str,
    query: str,
) -> tuple[dict, float]:
    atoms_json = json.dumps(
        [
            {
                "node_id": a["node_id"],
                "predicate": a["predicate"],
                "arg1": a["arg1"],
                "arg2": a.get("arg2", ""),
            }
            for a in l1_atoms[:20]
        ],
        indent=2,
    )
    prompt = L2_PROMPT.format(atoms_json=atoms_json, query=query)
    resp, cost = await llm_call(prompt, max_tokens=600)
    assumptions = extract_json(resp)
    if not isinstance(assumptions, dict):
        logger.warning(f"L2 parse failed: {resp[:100]}")
        assumptions = {"unifications": [], "bridge_rules": []}
    assumptions.setdefault("unifications", [])
    assumptions.setdefault("bridge_rules", [])

    clean_uni = []
    for u in assumptions["unifications"]:
        if isinstance(u, dict) and u.get("atom1") and u.get("atom2"):
            clean_uni.append({
                "atom1": str(u.get("atom1", "")),
                "atom2": str(u.get("atom2", "")),
                "weight": float(u.get("weight", 0.5)),
                "rationale": str(u.get("rationale", "")),
            })
    clean_br = []
    for b in assumptions["bridge_rules"]:
        if isinstance(b, dict) and b.get("consequent"):
            ants = b.get("antecedent", [])
            if isinstance(ants, str):
                ants = [ants]
            clean_br.append({
                "antecedent": [str(a) for a in ants],
                "consequent": str(b.get("consequent", "")),
                "weight": float(b.get("weight", 0.5)),
                "rationale": str(b.get("rationale", "")),
            })
    return {"unifications": clean_uni, "bridge_rules": clean_br}, cost


# ═══════════════════════════════════════════════════════════════════════════════
# ATMS PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def build_atms(
    l1_atoms: list[dict],
    l2: dict,
    query_node: str,
    domain: str = "clutrr",
) -> BoundedATMS:
    atms = BoundedATMS(depth_cap=DEPTH_CAP, beam_width=BEAM_WIDTH)

    for atom in l1_atoms:
        if atom.get("node_id"):
            atms.add_fact(atom["node_id"], env=frozenset())

    atms._get_or_create(query_node)

    assumption_counter = [0]

    def new_assumption_id() -> str:
        assumption_counter[0] += 1
        return f"a{assumption_counter[0]}"

    for u in l2.get("unifications", []):
        a_id = new_assumption_id()
        a = Assumption(id=a_id, text=f"{u['atom1']}={u['atom2']}", weight=u["weight"])
        atms.add_assumption(a)
        atms._get_or_create(u["atom1"])
        atms._get_or_create(u["atom2"])
        atms.merge_nodes(u["atom1"], u["atom2"], via_assumption=a_id)

    for b in l2.get("bridge_rules", []):
        a_id = new_assumption_id()
        a = Assumption(id=a_id, text=b["consequent"], weight=b["weight"])
        atms.add_assumption(a)
        for ant in b["antecedent"]:
            atms._get_or_create(ant)
        atms.add_rule(b["antecedent"], b["consequent"], via_assumption=a_id)

    if domain in ("clutrr", "custom", "ruletaker", "proofwriter"):
        _apply_kinship_rules(atms)
    else:
        _apply_location_rules(atms)

    atms.propagate()
    return atms


def score_conclusion(atms: BoundedATMS, query_node: str) -> dict:
    load = atms.assumption_load(query_node)
    is_empty = atms.is_empty_env_derivable(query_node)
    envs = atms.nodes.get(query_node)
    return {
        "assumption_load": load,
        "empty_env_derivable": is_empty,
        "minimal_environments": [list(e) for e in (envs.label if envs else set())],
        "derivable": load < float("inf"),
    }


def method_answer(score: dict) -> str:
    if not score["derivable"]:
        return "abstain"
    if score["assumption_load"] <= ASSUMPTION_THRESHOLD:
        return "yes"
    return "abstain"


def method_answer_with_fallback(score: dict) -> str:
    ans = method_answer(score)
    if ans == "abstain" and not score["derivable"]:
        return "no"
    return ans


# ═══════════════════════════════════════════════════════════════════════════════
# BASELINES WITH CONFIDENCE
# ═══════════════════════════════════════════════════════════════════════════════

COT_PROMPT_WITH_CONF = """\
Document: {document}

Question: {query}

Think step by step using ONLY information in the document.
End your response with:
  Answer: yes|no|unknown
  Confidence: 0.0-1.0 (your certainty that your answer is correct)
"""

LINC_PROMPT = """\
Document: {document}

Question: {query}

Reason formally: list the logical steps from document facts to the answer.
Then answer YES or NO based solely on document facts + basic logic.
End with: "Conclusion: yes" or "Conclusion: no" or "Conclusion: unknown"."""

PROBLOG_PROMPT = """\
Document: {document}

Question: {query}

List each explicit fact from the document as: fact(confidence).
Example: parent(alice,bob)(0.99). sibling(bob,carol)(0.99).
Then state if the answer is derivable.
End with: "Derivable: yes" or "Derivable: no"."""


def parse_yes_no(text: str, patterns: list[str] | None = None) -> str:
    if patterns is None:
        patterns = ["answer:", "conclusion:", "result:", "derivable:"]
    text_lower = text.lower()
    for pat in patterns:
        idx = text_lower.rfind(pat)
        if idx != -1:
            after = text_lower[idx + len(pat):idx + len(pat) + 30]
            if "yes" in after:
                return "yes"
            if "no" in after:
                return "no"
    last_yes = text_lower.rfind("yes")
    last_no = text_lower.rfind("no")
    if last_yes == -1 and last_no == -1:
        return "unknown"
    if last_yes > last_no:
        return "yes"
    return "no"


def _extract_confidence(text: str) -> float:
    m = re.search(r'confidence:\s*([0-9]*\.?[0-9]+)', text, re.IGNORECASE)
    if m:
        v = float(m.group(1))
        return min(max(v, 0.0), 1.0)
    return 0.5


async def cot_baseline_v2(document: str, query: str) -> tuple[str, float, float]:
    """Returns (answer, cost, confidence)."""
    prompt = COT_PROMPT_WITH_CONF.format(document=document[:2000], query=query)
    resp, cost = await llm_call(prompt, max_tokens=450)
    answer = parse_yes_no(resp)
    conf = _extract_confidence(resp)
    return answer, cost, conf


async def linc_sample(document: str, query: str) -> tuple[str, float]:
    prompt = LINC_PROMPT.format(document=document[:2000], query=query)
    resp, cost = await llm_call(prompt, max_tokens=350)
    answer = parse_yes_no(resp, patterns=["conclusion:", "answer:"])
    return answer, cost


async def linc_baseline_v2(document: str, query: str, k: int = LINC_K) -> tuple[str, float, float]:
    """Returns (majority_answer, cost, confidence)."""
    tasks = [linc_sample(document, query) for _ in range(k)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    answers, total_cost = [], 0.0
    for r in results:
        if isinstance(r, Exception):
            logger.warning(f"LINC sample failed: {r}")
            continue
        ans, cost = r
        answers.append(ans)
        total_cost += cost
    if not answers:
        return "unknown", total_cost, 0.0
    counter = Counter(a for a in answers if a != "unknown")
    if not counter:
        return "unknown", total_cost, 0.0
    top_ans, top_count = counter.most_common(1)[0]
    conf = top_count / len(answers)
    return top_ans, total_cost, conf


async def problog_baseline(document: str, query: str) -> tuple[str, float, float]:
    """Returns (answer, cost, derived_prob)."""
    prompt = PROBLOG_PROMPT.format(document=document[:2000], query=query)
    resp, cost = await llm_call(prompt, max_tokens=400)
    answer = parse_yes_no(resp, patterns=["derivable:", "answer:", "conclusion:"])
    nums = re.findall(r'\b(0\.\d+|1\.0)\b', resp)
    prob = float(nums[0]) if nums else (1.0 if answer == "yes" else 0.0)
    return answer, cost, prob


async def run_baselines_v2(inst: "Instance") -> dict:
    """Run all baselines with confidence scores."""
    cot_ans, cot_cost, cot_conf = await cot_baseline_v2(inst.document, inst.query)
    linc_ans, linc_cost, linc_conf = await linc_baseline_v2(inst.document, inst.query, k=LINC_K)
    problog_ans, pb_cost, pb_prob = await problog_baseline(inst.document, inst.query)
    return {
        "cot": cot_ans,
        "cot_confidence": cot_conf,
        "linc": linc_ans,
        "linc_confidence": linc_conf,
        "problog": problog_ans,
        "problog_prob": pb_prob,
        "cot_cost": cot_cost,
        "linc_cost": linc_cost,
        "problog_cost": pb_cost,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET LOADING
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Instance:
    doc_id: str
    document: str
    query: str
    gold_answer: str  # "yes", "no", or "unknown"
    hop_count: int
    source: str  # "clutrr", "ruletaker", "proofwriter"
    query_node: str


def _build_query_node(query: str, document: str) -> str:
    patterns = [
        r"([\w]+)\s+is\s+([\w]+)[\"']?s\s+([\w_]+)",
        r'what\s+is\s+the\s+([\w_]+)\s+of\s+([\w]+)\s+to\s+([\w]+)',
        r'([\w]+)\s+and\s+([\w]+)\s+are\s+([\w_]+)',
        r'is\s+([\w]+)\s+the\s+([\w_]+)\s+of\s+([\w]+)',
    ]
    for pat in patterns:
        m = re.search(pat, query, re.IGNORECASE)
        if m:
            groups = [g.lower() for g in m.groups()]
            if len(groups) >= 3:
                return f"relation({groups[0]},{groups[2]})"
            elif len(groups) == 2:
                return f"relation({groups[0]},{groups[1]})"
    return f"query_{abs(hash(query)) % 10000}"


def _build_query_node_explicit(query: str, document: str) -> str:
    m = re.match(r'[Tt]he\s+(\w+)\s+is\s+(\w+)', query)
    if m:
        return f"{m.group(2).lower()}({m.group(1).lower()})"
    m = re.match(r'[Tt]he\s+(\w+)\s+(\w+)s?\s+the\s+(\w+)', query)
    if m:
        return f"{m.group(2).lower()}({m.group(1).lower()},{m.group(3).lower()})"
    return f"query_{abs(hash(query)) % 10000}"


def _extract_persons_from_clutrr(query: str, story: str) -> tuple[str, str] | None:
    m = re.search(r'between\s+\[?(\w+)\]?\s+and\s+\[?(\w+)\]?', query, re.IGNORECASE)
    if m:
        return m.group(1).lower(), m.group(2).lower()
    names = re.findall(r'\b([A-Z][a-z]+)\b', query)
    if len(names) >= 2:
        return names[-2].lower(), names[-1].lower()
    return None


def load_full_dataset() -> list[Instance]:
    """Load 380 instances from full_data_out.json dependency."""
    if not DATASET_PATH.exists():
        logger.error(f"Dataset not found at {DATASET_PATH}")
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    data = json.loads(DATASET_PATH.read_text())
    instances = []

    for ds_block in data["datasets"]:
        src = ds_block["dataset"].lower()  # "clutrr", "ruletaker", "proofwriter"
        for ex in ds_block["examples"]:
            input_text = ex["input"]
            raw_output = ex["output"]
            hop = int(ex.get("metadata_hop_count", 1))
            meta_id = ex.get("metadata_id", "")

            # Split input into document + query
            if "\n\nQuestion:" in input_text:
                ctx_part, q_part = input_text.split("\n\nQuestion:", 1)
                document = ctx_part.replace("Context:", "").strip()
                query_raw = q_part.strip()
            else:
                document = input_text
                query_raw = input_text[-200:]

            # Normalize gold to binary yes/no
            if src in ("ruletaker", "proofwriter"):
                gold = {"true": "yes", "false": "no", "unknown": "unknown"}.get(
                    raw_output.lower(), "unknown"
                )
                query_text = query_raw
                query_node = _build_query_node_explicit(query_raw, document)
            else:  # clutrr
                persons = _extract_persons_from_clutrr(query_raw, document)
                if persons:
                    p1, p2 = persons
                    query_text = f"Is {p2} the {raw_output} of {p1}?"
                else:
                    query_text = query_raw
                gold = "yes"
                query_node = _build_query_node(query_text, document)

            instances.append(Instance(
                doc_id=meta_id or f"{src}_{len(instances)}",
                document=document,
                query=query_text,
                gold_answer=gold,
                hop_count=hop,
                source=src,
                query_node=query_node,
            ))

    logger.info(f"Loaded {len(instances)} instances from full_data_out.json")
    return instances


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 0: TRACTABILITY PILOT
# ═══════════════════════════════════════════════════════════════════════════════

async def run_single_instance(inst: Instance) -> dict:
    """Full ATMS pipeline for one instance."""
    t0 = time.perf_counter()

    l1_atoms, l1_cost = await extract_l1_atoms(inst.document)
    l2, l2_cost = await propose_l2_assumptions(l1_atoms, inst.document, inst.query)

    domain = inst.source  # "clutrr", "ruletaker", "proofwriter"
    atms = build_atms(l1_atoms, l2, inst.query_node, domain=domain)

    score = score_conclusion(atms, inst.query_node)
    our_ans = method_answer(score)
    our_ans_eval = method_answer_with_fallback(score)

    n_choice_points = len(l2.get("unifications", [])) + len(l2.get("bridge_rules", []))
    wall_clock = time.perf_counter() - t0

    return {
        "doc_id": inst.doc_id,
        "source": inst.source,
        "document": inst.document,
        "query": inst.query,
        "gold": inst.gold_answer,
        "hop_count": inst.hop_count,
        "l1_atoms": l1_atoms,
        "l2": l2,
        "n_choice_points": n_choice_points,
        "n_nogoods": len(atms.nogoods),
        "wall_clock_sec": wall_clock,
        "assumption_load": score["assumption_load"],
        "empty_env_derivable": score["empty_env_derivable"],
        "minimal_envs": score["minimal_environments"],
        "our_method": our_ans,
        "our_method_eval": our_ans_eval,
        "l1_cost": l1_cost,
        "l2_cost": l2_cost,
        "total_cost_this": l1_cost + l2_cost,
    }


async def stage0_pilot(pilot_instances: list[Instance]) -> dict:
    logger.info(f"Stage 0: tractability pilot on {len(pilot_instances)} docs")
    tasks = [run_single_instance(inst) for inst in pilot_instances]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    valid = [r for r in results if isinstance(r, dict)]
    if not valid:
        return {"decision": "tractability_failure", "mean_choice_points": 0, "mean_wall_clock_sec": 999}

    mean_cp = float(np.mean([r["n_choice_points"] for r in valid]))
    mean_wc = float(np.mean([r["wall_clock_sec"] for r in valid]))
    mean_ng = float(np.mean([r["n_nogoods"] for r in valid]))
    mean_cost = float(np.mean([r["total_cost_this"] for r in valid]))

    logger.info(f"Stage 0: mean_cp={mean_cp:.1f} mean_wc={mean_wc:.2f}s mean_cost=${mean_cost:.5f}")

    global DEPTH_CAP, BEAM_WIDTH
    if mean_wc < 30.0 and mean_cp <= 20:
        decision = "exact_atms"
    elif mean_wc < 90.0:
        decision = "exact_atms_reduced"
        DEPTH_CAP = 2
        BEAM_WIDTH = 10
        logger.warning("Reducing depth_cap=2, beam_width=10 for tractability")
    else:
        decision = "tractability_failure"
        logger.error("Stage 0: tractability failure (>90s/doc)")

    return {
        "mean_choice_points": mean_cp,
        "mean_nogoods": mean_ng,
        "mean_wall_clock_sec": mean_wc,
        "mean_llm_cost_per_doc": mean_cost,
        "decision": decision,
        "n_pilot_docs": len(valid),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN EXPERIMENT
# ═══════════════════════════════════════════════════════════════════════════════

async def process_instance_full_v2(inst: Instance) -> dict | None:
    """Run our method + all baselines (with confidence) on one instance."""
    global _cumulative_cost
    if _cumulative_cost >= COST_CAP:
        logger.warning(f"Cost cap reached at ${_cumulative_cost:.4f}, stopping")
        return None
    try:
        method_task = asyncio.create_task(run_single_instance(inst))
        baseline_task = asyncio.create_task(run_baselines_v2(inst))
        method_result, baseline_result = await asyncio.gather(method_task, baseline_task)
        result = {**method_result, **baseline_result}
        result["cumulative_cost"] = _cumulative_cost
        return result
    except Exception:
        logger.exception(f"Failed on {inst.doc_id}")
        return None


async def run_experiment_v2(instances: list[Instance], max_concurrent: int = 6) -> list[dict]:
    """Run full experiment with controlled concurrency."""
    results = []
    batch_size = max_concurrent

    logger.info(f"Running experiment on {len(instances)} instances (batch={batch_size})")
    for i in range(0, len(instances), batch_size):
        if _cumulative_cost >= COST_CAP:
            logger.warning("Cost cap reached, stopping experiment")
            break

        batch = instances[i:i + batch_size]
        tasks = [process_instance_full_v2(inst) for inst in batch]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

        for r in batch_results:
            if isinstance(r, dict):
                results.append(r)
            elif isinstance(r, Exception):
                logger.warning(f"Batch item exception: {r}")

        logger.info(f"Progress: {len(results)}/{len(instances)} | cost=${_cumulative_cost:.4f}")
        gc.collect()

    return results


async def run_70b_ablation(instances: list[Instance], n_subset: int = 150) -> list[dict]:
    """Run full pipeline on stratified 150-instance subset using LARGE_MODEL."""
    global _current_model

    # Build stratified subset
    by_src: dict[str, list[Instance]] = {}
    for inst in instances:
        by_src.setdefault(inst.source, []).append(inst)

    subset: list[Instance] = []
    per_src = n_subset // max(len(by_src), 1)
    for src, src_insts in by_src.items():
        subset.extend(src_insts[:per_src])
    subset = subset[:n_subset]

    logger.info(f"70B ablation: {len(subset)} instances using {LARGE_MODEL}")

    saved_model = _current_model
    _current_model = LARGE_MODEL

    results_70b = await run_experiment_v2(subset, max_concurrent=4)

    _current_model = saved_model
    logger.info(f"70B ablation done: {len(results_70b)} instances | total_cost=${_cumulative_cost:.4f}")
    return results_70b


# ═══════════════════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def _safe_mean(lst: list) -> float:
    return float(np.mean(lst)) if lst else float("nan")


def _safe_acc(pairs: list[tuple[str, str]]) -> float:
    valid = [(p, g) for p, g in pairs if p not in ("unknown", "abstain") and g not in ("unknown",)]
    return _safe_mean([int(p == g) for p, g in valid])


def compute_metrics(results: list[dict]) -> dict:
    all_results = results

    # H1: empty-env precision
    ee_results = [r for r in all_results if r.get("empty_env_derivable")]
    ee_precision = (
        _safe_mean([int(r.get("our_method_eval", r["our_method"]) == r["gold"]) for r in ee_results
                    if r["gold"] not in ("unknown",)])
        if ee_results else float("nan")
    )

    # L1 grounding error rate
    total_atoms = sum(len(r.get("l1_atoms", [])) for r in results)
    grounding_errors = sum(
        1 for r in results for a in r.get("l1_atoms", [])
        if not a.get("grounding_verified", True)
    )
    l1_error_rate = grounding_errors / max(total_atoms, 1)

    # H2: Spearman correlation
    finite_results = [r for r in all_results if r.get("assumption_load", float("inf")) < float("inf")]
    spearman_rho, spearman_p, auroc_load = float("nan"), float("nan"), float("nan")

    if len(finite_results) >= 10:
        loads = [r["assumption_load"] for r in finite_results]
        errors = [int(r.get("our_method_eval", r["our_method"]) != r["gold"]) for r in finite_results]
        try:
            rho, p = spearmanr(loads, errors)
            spearman_rho, spearman_p = float(rho), float(p)
        except Exception:
            pass
        if len(set(errors)) > 1:
            try:
                auroc_load = float(roc_auc_score(errors, loads))
            except Exception:
                pass

    # Abstain rate and answered accuracy
    abstain_rate = _safe_mean([int(r["our_method"] == "abstain") for r in all_results])
    answered = [r for r in all_results if r["our_method"] != "abstain" and r["gold"] not in ("unknown",)]
    answered_accuracy_our = _safe_mean([int(r["our_method"] == r["gold"]) for r in answered])

    # H4: implicit vs explicit (kinship vs rule-based)
    implicit = [r for r in all_results if r["source"] in ("clutrr",)]
    explicit = [r for r in all_results if r["source"] in ("ruletaker", "proofwriter")]

    def acc(subset: list[dict], key: str) -> float:
        eval_key = "our_method_eval" if key == "our_method" else key
        pairs = [(r.get(eval_key, r.get(key, "unknown")), r["gold"]) for r in subset]
        return _safe_acc(pairs)

    implicit_our_acc = acc(implicit, "our_method")
    implicit_cot_acc = acc(implicit, "cot")
    explicit_our_acc = acc(explicit, "our_method")
    explicit_cot_acc = acc(explicit, "cot")

    implicit_adv = (
        implicit_our_acc - implicit_cot_acc
        if len(implicit) > 0 and not math.isnan(implicit_our_acc) and not math.isnan(implicit_cot_acc)
        else float("nan")
    )
    explicit_adv = (
        explicit_our_acc - explicit_cot_acc
        if len(explicit) > 0 and not math.isnan(explicit_our_acc) and not math.isnan(explicit_cot_acc)
        else float("nan")
    )

    # Overall accuracy per method
    def overall_acc(key: str) -> float:
        eval_key = "our_method_eval" if key == "our_method" else key
        pairs = [(r.get(eval_key, r.get(key, "unknown")), r["gold"]) for r in all_results]
        return _safe_acc(pairs)

    # Risk-coverage AUC for ATMS
    def risk_coverage_auc_atms(results_list: list[dict]) -> float:
        scores = [r.get("assumption_load", float("inf")) for r in results_list]
        finite_scores = [s for s in scores if s < float("inf")]
        if not finite_scores:
            return float("nan")
        thresholds = sorted(set(finite_scores))
        areas = []
        prev_cov = 0.0
        for t in thresholds:
            answered_here = [
                r for r, s in zip(results_list, scores)
                if s <= t and r["gold"] not in ("unknown",)
            ]
            if not answered_here:
                continue
            coverage = len(answered_here) / max(len(results_list), 1)
            risk = _safe_mean([int(r["our_method_eval"] != r["gold"]) for r in answered_here])
            areas.append((coverage - prev_cov) * risk)
            prev_cov = coverage
        return float(sum(areas)) if areas else float("nan")

    def rc_auc_baseline(results_list: list[dict], key: str) -> float:
        answered = [r for r in results_list if r.get(key, "unknown") not in ("unknown",) and r["gold"] not in ("unknown",)]
        if not answered:
            return float("nan")
        n = max(len(results_list), 1)
        coverage = len(answered) / n
        risk = _safe_mean([int(r[key] != r["gold"]) for r in answered])
        return risk * coverage

    return {
        "l1_grounding_error_rate": l1_error_rate,
        "empty_env_precision": ee_precision,
        "n_empty_env_derivable": len(ee_results),
        "spearman_rho": spearman_rho,
        "spearman_p": spearman_p,
        "auroc_assumption_load": auroc_load,
        "risk_coverage_auc": {
            "our_method": risk_coverage_auc_atms(all_results),
            "cot": rc_auc_baseline(all_results, "cot"),
            "linc": rc_auc_baseline(all_results, "linc"),
            "problog": rc_auc_baseline(all_results, "problog"),
        },
        "implicit_advantage_gap": implicit_adv,
        "explicit_advantage_gap": explicit_adv,
        "abstain_rate": abstain_rate,
        "answered_accuracy_our_method": answered_accuracy_our,
        "overall_accuracy": {
            "our_method": overall_acc("our_method"),
            "cot": overall_acc("cot"),
            "linc": overall_acc("linc"),
            "problog": overall_acc("problog"),
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
# ITER-2 ANALYSES
# ═══════════════════════════════════════════════════════════════════════════════

def compute_matched_coverage_baselines(
    results: list[dict],
    target_coverage: float = TARGET_COVERAGE,
) -> dict:
    """
    For each baseline, find confidence threshold θ such that
    fraction of instances with confidence >= θ equals target_coverage.
    Compare answered-accuracy with ATMS at same coverage.
    """
    n = len(results)
    if n == 0:
        return {}
    target_n = max(1, round(target_coverage * n))

    out = {}
    for baseline_key, conf_key in [
        ("cot", "cot_confidence"),
        ("linc", "linc_confidence"),
        ("problog", "problog_prob"),
    ]:
        confs = np.array([r.get(conf_key, 0.5) for r in results])
        preds = [r.get(baseline_key, "unknown") for r in results]
        golds = [r["gold"] for r in results]

        order = np.argsort(-confs)
        top_indices = order[:target_n]

        answered_preds = [preds[i] for i in top_indices]
        answered_golds = [golds[i] for i in top_indices]

        valid_pairs = [
            (p, g) for p, g in zip(answered_preds, answered_golds)
            if p not in ("unknown",) and g not in ("unknown",)
        ]
        acc = _safe_mean([int(p == g) for p, g in valid_pairs]) if valid_pairs else float("nan")

        out[baseline_key] = {
            "target_coverage": target_coverage,
            "actual_coverage": len(top_indices) / n,
            "n_answered": len(top_indices),
            "accuracy_at_matched_coverage": acc,
            "threshold_used": float(confs[order[target_n - 1]]) if target_n > 0 and len(order) >= target_n else float("nan"),
        }

    # ATMS matched coverage: top-target_n instances by LOWEST assumption_load
    loads = np.array([r.get("assumption_load", float("inf")) for r in results])
    finite_mask = np.isfinite(loads)
    n_finite = int(finite_mask.sum())

    if n_finite >= target_n:
        finite_indices = np.where(finite_mask)[0]
        sorted_finite = finite_indices[np.argsort(loads[finite_indices])]
        atms_top = sorted_finite[:target_n]
    else:
        atms_top = np.where(finite_mask)[0]

    atms_preds = [results[i].get("our_method_eval", "unknown") for i in atms_top]
    atms_golds = [results[i]["gold"] for i in atms_top]
    atms_valid = [(p, g) for p, g in zip(atms_preds, atms_golds)
                  if p not in ("unknown", "abstain") and g not in ("unknown",)]
    atms_acc = _safe_mean([int(p == g) for p, g in atms_valid]) if atms_valid else float("nan")

    out["atms_matched_coverage"] = {
        "target_coverage": target_coverage,
        "actual_coverage": len(atms_top) / n,
        "n_answered": len(atms_top),
        "accuracy_at_matched_coverage": atms_acc,
    }

    return out


def compute_random_oracle(
    results: list[dict],
    target_coverage: float = TARGET_COVERAGE,
    n_seeds: int = RANDOM_ORACLE_SEEDS,
) -> dict:
    """Random abstention oracle: randomly sample target_coverage fraction, average accuracy."""
    n = len(results)
    if n == 0:
        return {"target_coverage": target_coverage, "n_seeds": n_seeds, "mean_accuracy": float("nan"), "std_accuracy": float("nan")}
    k = max(1, round(target_coverage * n))
    accs = []
    for seed in range(n_seeds):
        random.seed(seed)
        indices = random.sample(range(n), min(k, n))
        acc_vals = []
        for i in indices:
            r = results[i]
            pred = r.get("our_method_eval", "unknown")
            if pred not in ("unknown", "abstain") and r["gold"] not in ("unknown",):
                acc_vals.append(int(pred == r["gold"]))
        if acc_vals:
            accs.append(float(np.mean(acc_vals)))
    mean_acc = _safe_mean(accs)
    std_acc = float(np.std(accs)) if accs else float("nan")
    return {
        "target_coverage": target_coverage,
        "n_seeds": n_seeds,
        "mean_accuracy": mean_acc,
        "std_accuracy": std_acc,
    }


def stratified_h2_analysis(results: list[dict]) -> dict:
    """Compute per-stratum error rate and Spearman rho for H2 monotonicity."""
    strata: dict[str, list[dict]] = {"0": [], "1": [], "2": [], "3+": [], "inf": []}
    for r in results:
        load = r.get("assumption_load", float("inf"))
        if not isinstance(load, (int, float)) or math.isnan(load):
            strata["inf"].append(r)
        elif math.isinf(load):
            strata["inf"].append(r)
        elif load == 0:
            strata["0"].append(r)
        elif load == 1:
            strata["1"].append(r)
        elif load == 2:
            strata["2"].append(r)
        else:
            strata["3+"].append(r)

    out: dict = {}
    for k, group in strata.items():
        if not group:
            out[k] = {"n": 0, "error_rate": float("nan"), "note": "empty stratum"}
            continue
        valid = [
            r for r in group
            if r.get("our_method_eval") not in ("unknown", "abstain")
            and r["gold"] not in ("unknown",)
        ]
        error_rate = _safe_mean([int(r["our_method_eval"] != r["gold"]) for r in valid]) if valid else float("nan")
        out[k] = {
            "n": len(group),
            "n_evaluable": len(valid),
            "error_rate": error_rate,
            "h2_testable": len(group) >= 30,
        }

    # Spearman on finite-load instances
    finite = [r for r in results if r.get("assumption_load", float("inf")) < float("inf")
              and not math.isnan(r.get("assumption_load", float("nan")))]
    spearman_rho, spearman_p = float("nan"), float("nan")
    if len(finite) >= 10:
        loads = [r["assumption_load"] for r in finite]
        errors = [int(r.get("our_method_eval", "unknown") != r["gold"]) for r in finite]
        try:
            rho, p = spearmanr(loads, errors)
            spearman_rho, spearman_p = float(rho), float(p)
        except Exception:
            pass
    out["spearman_rho"] = spearman_rho
    out["spearman_p"] = spearman_p
    out["n_finite_load"] = len(finite)
    return out


def atms_vs_problog_divergence(results: list[dict]) -> dict:
    """Annotate instances where ATMS and ProbLog disagree; compare against gold."""
    divergent = [
        r for r in results
        if r.get("problog") not in ("unknown", None)
        and r.get("our_method_eval") not in ("unknown", "abstain", None)
        and r.get("problog") != r.get("our_method_eval")
        and r["gold"] not in ("unknown",)
    ]
    records = []
    for r in divergent:
        atms_correct = int(r["our_method_eval"] == r["gold"])
        problog_correct = int(r["problog"] == r["gold"])
        records.append({
            "doc_id": r["doc_id"],
            "assumption_load": r.get("assumption_load"),
            "atms_answer": r["our_method_eval"],
            "problog_answer": r["problog"],
            "gold": r["gold"],
            "atms_correct": atms_correct,
            "problog_correct": problog_correct,
        })
    n = len(results)
    n_div = len(divergent)
    atms_wins = sum(1 for rec in records if rec["atms_correct"] and not rec["problog_correct"])
    problog_wins = sum(1 for rec in records if rec["problog_correct"] and not rec["atms_correct"])
    both_wrong = sum(1 for rec in records if not rec["atms_correct"] and not rec["problog_correct"])
    return {
        "n_total": n,
        "n_divergent": n_div,
        "divergence_rate": n_div / max(n, 1),
        "atms_wins_on_divergent": atms_wins,
        "problog_wins_on_divergent": problog_wins,
        "both_wrong_on_divergent": both_wrong,
        "divergent_records": records,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# OUTPUT FORMATTING
# ═══════════════════════════════════════════════════════════════════════════════

def format_result_for_schema(results: list[dict]) -> dict:
    """Format results into exp_gen_sol_out schema."""
    by_source: dict[str, list[dict]] = {}
    for r in results:
        by_source.setdefault(r["source"], []).append(r)

    datasets = []
    for src, src_results in by_source.items():
        examples = []
        for r in src_results:
            input_str = f"Document: {r.get('document', r.get('query', ''))[:500]}\n\nQuery: {r['query']}"
            al = r.get("assumption_load", float("inf"))
            al_str = "inf" if al == float("inf") or (isinstance(al, float) and math.isinf(al)) else str(al)

            example = {
                "input": input_str,
                "output": r["gold"],
                "predict_our_method": r.get("our_method_eval", r.get("our_method", "unknown")),
                "predict_cot": r.get("cot", "unknown"),
                "predict_linc": r.get("linc", "unknown"),
                "predict_problog": r.get("problog", "unknown"),
                "metadata_hop_count": str(r.get("hop_count", 0)),
                "metadata_assumption_load": al_str,
                "metadata_empty_env_derivable": str(r.get("empty_env_derivable", False)),
                "metadata_n_choice_points": str(r.get("n_choice_points", 0)),
                "metadata_cot_confidence": str(round(r.get("cot_confidence", 0.5), 4)),
                "metadata_linc_confidence": str(round(r.get("linc_confidence", 0.5), 4)),
                "metadata_problog_prob": str(round(r.get("problog_prob", 0.5), 4)),
                "metadata_cumulative_cost_usd": str(round(r.get("cumulative_cost", 0), 5)),
                "metadata_wall_clock_sec": str(round(r.get("wall_clock_sec", 0), 3)),
            }
            examples.append(example)
        datasets.append({"dataset": src, "examples": examples})

    return {"datasets": datasets}


def _nan_audit(d: dict, path: str = "") -> list[str]:
    """Recursively find NaN/Inf float fields and return their paths."""
    issues = []
    for k, v in d.items():
        full_path = f"{path}.{k}" if path else k
        if isinstance(v, float):
            if math.isnan(v):
                issues.append(f"NaN at {full_path}")
            elif math.isinf(v):
                issues.append(f"Inf at {full_path}")
        elif isinstance(v, dict):
            issues.extend(_nan_audit(v, full_path))
    return issues


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

@logger.catch(reraise=True)
async def main() -> None:
    logger.info("=== ATMS Iter-2 Experiment ===")
    logger.info(f"Model: {CHEAP_MODEL} | Cost cap: ${COST_CAP}")

    if not OPENROUTER_KEY:
        logger.error("OPENROUTER_API_KEY not set")
        sys.exit(1)

    # Step 1: Load dataset
    instances = load_full_dataset()
    eval_instances = [i for i in instances if i.gold_answer not in ("unknown",)]
    logger.info(f"Total: {len(instances)} | Eval (non-unknown gold): {len(eval_instances)}")

    # Step 2: Stage-0 tractability pilot (first 15 eval instances)
    pilot = eval_instances[:15]
    stage0 = await stage0_pilot(pilot)
    logger.info(f"Stage 0: {stage0['decision']}")

    if stage0["decision"] == "tractability_failure":
        out = {"metadata": {"stage0_tractability": stage0, "error": "tractability_failure"}, "datasets": []}
        out_path = WORKSPACE / "full_method_out.json"
        out_path.write_text(json.dumps(out, indent=2, default=str))
        logger.error(f"Tractability failure — partial output written to {out_path}")
        await close_session()
        return

    # Step 3: Main experiment
    results = await run_experiment_v2(eval_instances, max_concurrent=6)
    logger.info(f"Main experiment: {len(results)} results | cost=${_cumulative_cost:.4f}")

    # Step 4: 70B ablation (only if remaining budget > $3)
    results_70b: list[dict] = []
    budget_remaining = COST_CAP - _cumulative_cost
    if budget_remaining > 3.0:
        logger.info(f"Running 70B ablation (budget remaining: ${budget_remaining:.2f})")
        results_70b = await run_70b_ablation(eval_instances, n_subset=150)
    else:
        logger.warning(f"Skipping 70B ablation: insufficient budget (${budget_remaining:.2f} remaining)")

    # Step 5: Compute all metrics
    metrics = compute_metrics(results)
    matched_cov = compute_matched_coverage_baselines(results)
    random_oracle = compute_random_oracle(results)
    strat = stratified_h2_analysis(results)
    div = atms_vs_problog_divergence(results)
    metrics_70b = compute_metrics(results_70b) if results_70b else {}

    # Log key metrics
    logger.info(f"Spearman rho={metrics['spearman_rho']:.3f} p={metrics['spearman_p']:.3f}")
    logger.info(f"EE precision={metrics['empty_env_precision']:.3f} n={metrics['n_empty_env_derivable']}")
    logger.info(f"Overall acc: ATMS={metrics['overall_accuracy']['our_method']:.3f} CoT={metrics['overall_accuracy']['cot']:.3f}")
    logger.info(f"Abstain rate={metrics['abstain_rate']:.3f}")
    logger.info(f"ATMS matched-cov accuracy={matched_cov.get('atms_matched_coverage', {}).get('accuracy_at_matched_coverage', 'N/A')}")

    # Step 6: Build output
    schema_out = format_result_for_schema(results)
    schema_out["metadata"] = {
        "method_name": "Bounded-ATMS-v2",
        "description": "Iter-2: matched-coverage baselines, 70B ablation, stratification, ATMS-vs-ProbLog ground-truth",
        "model": CHEAP_MODEL,
        "n_instances": len(results),
        "n_instances_70b_ablation": len(results_70b),
        "stage0_tractability": stage0,
        "total_cost_usd": _cumulative_cost,
        "parameters": {
            "depth_cap": DEPTH_CAP,
            "beam_width": BEAM_WIDTH,
            "assumption_threshold": ASSUMPTION_THRESHOLD,
            "linc_k": LINC_K,
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
            "model": LARGE_MODEL,
            "n_instances": len(results_70b),
            "metrics": metrics_70b,
        },
    }

    # NaN audit
    nan_issues = _nan_audit(schema_out.get("metadata", {}))
    if nan_issues:
        logger.warning(f"NaN/Inf in metadata: {nan_issues}")

    # Save
    out_path = WORKSPACE / "full_method_out.json"
    out_path.write_text(json.dumps(schema_out, indent=2, default=str))
    size_mb = out_path.stat().st_size / 1024 / 1024
    logger.info(f"Saved: {out_path} ({size_mb:.2f} MB)")

    # JSON round-trip check
    try:
        _ = json.loads(out_path.read_text())
        logger.info("JSON round-trip check: PASSED")
    except json.JSONDecodeError as e:
        logger.error(f"JSON round-trip check FAILED: {e}")

    await close_session()
    logger.info(f"=== Done. Total cost: ${_cumulative_cost:.4f} ===")


if __name__ == "__main__":
    asyncio.run(main())
