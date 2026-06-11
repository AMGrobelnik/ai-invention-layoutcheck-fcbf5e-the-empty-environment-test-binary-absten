#!/usr/bin/env python3
"""
Bounded ATMS Hallucination Meter for Text-to-Logic Reasoning.
Two-layer neuro-symbolic pipeline separating document-grounding (L1) from
assumption-tracking (L2) via a bounded Assumption-based Truth Maintenance System.
"""

from __future__ import annotations

import asyncio
import gc
import json
import math
import os
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
import pandas as pd
from loguru import logger
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from tenacity import retry, stop_after_attempt, wait_exponential

# ─── Workspace ────────────────────────────────────────────────────────────────
WORKSPACE = Path(__file__).parent
RESULTS_DIR = WORKSPACE / "results"
LOGS_DIR = WORKSPACE / "logs"
DATA_DIR = WORKSPACE / "data"
for d in (RESULTS_DIR, LOGS_DIR, DATA_DIR):
    d.mkdir(exist_ok=True)

# ─── Logging ──────────────────────────────────────────────────────────────────
logger.remove()
GREEN, CYAN, END = "\033[92m", "\033[96m", "\033[0m"
logger.add(
    sys.stdout,
    level="INFO",
    format=f"{GREEN}{{time:HH:mm:ss}}{END}|{{level:<7}}|{CYAN}{{function}}{END}| {{message}}",
)
logger.add(str(LOGS_DIR / "run.log"), rotation="30 MB", level="DEBUG")

# ─── Config ───────────────────────────────────────────────────────────────────
OPENROUTER_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE = "https://openrouter.ai/api/v1"
CHEAP_MODEL = "meta-llama/llama-3.1-8b-instruct"
FALLBACK_MODELS = ["google/gemini-flash-1.5-8b", "mistralai/mistral-7b-instruct"]
COST_CAP = 9.0
ASSUMPTION_THRESHOLD = 2  # load < threshold → answer yes, else abstain
DEPTH_CAP = 3
BEAM_WIDTH = 20
LINC_K = 3

# Memory limit: 8 GB virtual (experiment is lightweight, all LLM-based)
try:
    resource.setrlimit(resource.RLIMIT_AS, (8 * 1024**3, 8 * 1024**3))
except ValueError:
    pass  # may be lower than current; ignore

# ─── Cost Tracking ────────────────────────────────────────────────────────────
_cumulative_cost: float = 0.0

PRICE_PER_1M = {
    "meta-llama/llama-3.1-8b-instruct": {"in": 0.06, "out": 0.06},
    "google/gemini-flash-1.5-8b": {"in": 0.0375, "out": 0.15},
    "mistralai/mistral-7b-instruct": {"in": 0.055, "out": 0.055},
    "anthropic/claude-haiku-4-5": {"in": 0.80, "out": 4.0},
}

_current_model = CHEAP_MODEL
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
        timeout = aiohttp.ClientTimeout(total=90)
        _session = aiohttp.ClientSession(timeout=timeout)
    return _session


async def get_semaphore() -> asyncio.Semaphore:
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(8)  # 8 concurrent LLM calls
    return _semaphore


# ─── LLM Caller ──────────────────────────────────────────────────────────────
async def llm_call(prompt: str, max_tokens: int = 600, retries: int = 3) -> tuple[str, float]:
    """Call OpenRouter LLM. Returns (response_text, cost_usd)."""
    global _cumulative_cost, _current_model

    if _cumulative_cost >= COST_CAP:
        raise RuntimeError(f"Cost cap ${COST_CAP} reached: ${_cumulative_cost:.4f}")

    session = await get_session()
    sem = await get_semaphore()

    models_to_try = [_current_model] + [m for m in FALLBACK_MODELS if m not in _model_failed]

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
                        logger.debug(f"LLM [{model}] cost=${cost:.5f} total=${_cumulative_cost:.4f} | prompt={prompt[:80]!r} | resp={content[:80]!r}")
                        return content, cost
                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    logger.warning(f"LLM call attempt {attempt+1} failed: {e}")
                    await asyncio.sleep(2 * (attempt + 1))
            _model_failed.add(model)

    raise RuntimeError("All LLM models failed")


def extract_json(text: str) -> Any:
    """Robustly extract JSON from LLM response text."""
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try json5
    try:
        import json5
        return json5.loads(text)
    except Exception:
        pass
    # Regex extraction: find first JSON array or object
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
    weight: float  # 0-1, LLM-proposed confidence


@dataclass
class Node:
    id: str
    label: set  # set of frozenset[str] — minimal environments

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
        self._rules: list[tuple[list[str], str]] = []  # (antecedents, consequent)

    def _add_env_to_label(self, node_id: str, env: frozenset) -> bool:
        """Add env to node's label. Returns True if label changed."""
        node = self.nodes[node_id]
        # Check if env is already subsumed by an existing entry
        for existing in node.label:
            if existing <= env:  # existing is subset of env → env is redundant
                return False
        # Check depth cap
        if len(env) > self.depth_cap:
            return False
        # Check against nogoods
        for ng in self.nogoods:
            if ng <= env:
                return False
        # Remove any existing entries that are supersets of env (env subsumes them)
        to_remove = {e for e in node.label if env < e}
        node.label -= to_remove
        node.label.add(env)
        # Trim to beam_width: keep beam_width smallest (by size, then lex)
        if len(node.label) > self.beam_width:
            sorted_envs = sorted(node.label, key=lambda e: (len(e), sorted(e)))
            node.label = set(sorted_envs[: self.beam_width])
        return True

    def _get_or_create(self, node_id: str) -> Node:
        if node_id not in self.nodes:
            self.nodes[node_id] = Node(id=node_id, label=set())
        return self.nodes[node_id]

    def add_fact(self, node_id: str, env: frozenset | None = None) -> None:
        """Add a fact supported by env (default: empty = grounded)."""
        if env is None:
            env = frozenset()
        self._get_or_create(node_id)
        self._add_env_to_label(node_id, env)

    def add_assumption(self, assumption: Assumption) -> str:
        """Create an assumption node with singleton environment {assumption.id}."""
        self.assumptions[assumption.id] = assumption
        self._get_or_create(assumption.id)
        self._add_env_to_label(assumption.id, frozenset([assumption.id]))
        return assumption.id

    def add_rule(
        self,
        antecedents: list[str],
        consequent: str,
        via_assumption: str | None = None,
    ) -> None:
        """Register a rule and immediately forward-chain it."""
        rule_env_extra = frozenset([via_assumption]) if via_assumption else frozenset()
        self._get_or_create(consequent)
        for ant in antecedents:
            self._get_or_create(ant)
        self._rules.append((antecedents, consequent, rule_env_extra))
        self._fire_rule(antecedents, consequent, rule_env_extra)

    def _fire_rule(
        self,
        antecedents: list[str],
        consequent: str,
        rule_env_extra: frozenset,
    ) -> bool:
        """Derive new environments for consequent from antecedents."""
        changed = False
        # Collect all combinations of one env from each antecedent
        ant_labels = []
        for ant in antecedents:
            node = self.nodes.get(ant)
            if node is None or not node.label:
                return False
            ant_labels.append(list(node.label))

        # Enumerate Cartesian product (capped)
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
        """Register a nogood and propagate to prune labels."""
        self.nogoods.append(assumption_ids)
        # Prune all nodes
        for node in self.nodes.values():
            to_remove = {e for e in node.label if assumption_ids <= e}
            node.label -= to_remove

    def merge_nodes(self, node_id1: str, node_id2: str, via_assumption: str) -> None:
        """Unification: node1 and node2 share environments via an assumption."""
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
        """Re-fire all rules until fixpoint (max 10 iterations)."""
        for _ in range(10):
            changed = False
            for antecedents, consequent, rule_env_extra in self._rules:
                if self._fire_rule(antecedents, consequent, rule_env_extra):
                    changed = True
            if not changed:
                break

    def assumption_load(self, node_id: str) -> float:
        """Min size of minimal supporting env. 0=grounded, inf=not derivable."""
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
# ONTOLOGY
# ═══════════════════════════════════════════════════════════════════════════════

KINSHIP_ONTOLOGY = [
    # Type constraints: (antecedents, consequent)
    (["parent(X,Y)"], "person(X)"),
    (["parent(X,Y)"], "person(Y)"),
    (["sibling(X,Y)"], "person(X)"),
    (["sibling(X,Y)"], "person(Y)"),
    (["grandparent(X,Y)"], "person(X)"),
    (["grandparent(X,Y)"], "person(Y)"),
    # Transitivity & derived relations
    (["parent(X,Y)", "parent(Y,Z)"], "grandparent(X,Z)"),
    (["grandparent(X,Y)", "parent(Y,Z)"], "great_grandparent(X,Z)"),
    (["parent(X,Y)", "sibling(Y,Z)"], "aunt_or_uncle(X,Z)"),
    (["sibling(X,Y)"], "sibling(Y,X)"),  # symmetry
    (["parent(X,Y)"], "ancestor(X,Y)"),
    (["ancestor(X,Y)", "parent(Y,Z)"], "ancestor(X,Z)"),
]

LOCATION_ONTOLOGY = [
    # bAbI location rules
    (["traveled_to(X,L)"], "at(X,L)"),
    (["picked_up(X,O)"], "has(X,O)"),
    (["dropped(X,O)"], "not_has(X,O)"),
    (["at(X,L)", "at(Y,L)"], "same_location(X,Y)"),
]


def apply_ontology_to_atms(atms: BoundedATMS, domain: str) -> None:
    """Add ontology rules as L1 (empty-environment) rules."""
    rules = KINSHIP_ONTOLOGY if domain in ("clutrr", "custom") else LOCATION_ONTOLOGY
    for antecedents, consequent in rules:
        # These are schema-level rules — we apply them as templates
        # For document-specific facts, the actual variable bindings are in node IDs
        # We skip adding template rules (no bindings) and rely on L2 for instantiation
        pass  # Ontology applied during ATMS seeding with actual atoms


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
    """Extract grounded L1 atoms from document via LLM."""
    prompt = L1_PROMPT.format(document=document[:3000])
    resp, cost = await llm_call(prompt, max_tokens=800)
    atoms = extract_json(resp)
    if not isinstance(atoms, list):
        logger.warning(f"L1 parse failed, got: {resp[:100]}")
        atoms = []

    # Validate grounding: check span appears in document
    doc_lower = document.lower()
    for atom in atoms:
        if not isinstance(atom, dict):
            continue
        span = atom.get("span", "")
        atom["grounding_verified"] = span.lower() in doc_lower if span else False
        # Normalize
        for k in ("predicate", "arg1", "arg2"):
            atom.setdefault(k, "")
        atom["node_id"] = f"{atom['predicate']}({atom['arg1']},{atom['arg2']})" if atom.get("arg2") else f"{atom['predicate']}({atom['arg1']})"

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
    """Propose L2 assumptions (unifications + bridge rules)."""
    atoms_json = json.dumps([
        {"node_id": a["node_id"], "predicate": a["predicate"], "arg1": a["arg1"], "arg2": a.get("arg2", "")}
        for a in l1_atoms[:20]  # cap to avoid huge prompts
    ], indent=2)
    prompt = L2_PROMPT.format(atoms_json=atoms_json, query=query)
    resp, cost = await llm_call(prompt, max_tokens=600)
    assumptions = extract_json(resp)
    if not isinstance(assumptions, dict):
        logger.warning(f"L2 parse failed: {resp[:100]}")
        assumptions = {"unifications": [], "bridge_rules": []}
    assumptions.setdefault("unifications", [])
    assumptions.setdefault("bridge_rules", [])
    # Validate structure
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
    """Build and propagate ATMS from L1 atoms + L2 assumptions."""
    atms = BoundedATMS(depth_cap=DEPTH_CAP, beam_width=BEAM_WIDTH)

    # Seed L1 atoms as grounded (empty environment)
    for atom in l1_atoms:
        if atom.get("node_id"):
            atms.add_fact(atom["node_id"], env=frozenset())

    # Add query target node
    atms._get_or_create(query_node)

    # L2 unifications
    assumption_counter = [0]
    def new_assumption_id() -> str:
        assumption_counter[0] += 1
        return f"a{assumption_counter[0]}"

    for u in l2.get("unifications", []):
        a_id = new_assumption_id()
        a = Assumption(id=a_id, text=f"{u['atom1']}={u['atom2']}", weight=u["weight"])
        atms.add_assumption(a)
        # Ensure both nodes exist
        atms._get_or_create(u["atom1"])
        atms._get_or_create(u["atom2"])
        atms.merge_nodes(u["atom1"], u["atom2"], via_assumption=a_id)

    # L2 bridge rules
    for b in l2.get("bridge_rules", []):
        a_id = new_assumption_id()
        a = Assumption(id=a_id, text=b["consequent"], weight=b["weight"])
        atms.add_assumption(a)
        # Ensure antecedent nodes exist
        for ant in b["antecedent"]:
            atms._get_or_create(ant)
        atms.add_rule(b["antecedent"], b["consequent"], via_assumption=a_id)

    # Apply domain ontology rules to L1 atoms
    # Apply domain-specific ontology rules
    if domain in ("clutrr", "custom"):
        _apply_kinship_rules(atms)
    else:
        _apply_location_rules(atms)

    atms.propagate()
    return atms


# Canonical parent predicates (LLM may use any of these)
_PARENT_PREDS = re.compile(
    r'^(parent|father|mother|dad|mom|is_parent|is_father|is_mother|parent_of)\((.+?),(.+?)\)$',
    re.IGNORECASE,
)
_SIBLING_PREDS = re.compile(
    r'^(sibling|brother|sister|is_sibling|sibling_of)\((.+?),(.+?)\)$',
    re.IGNORECASE,
)
# Location predicates
_TRAVELED_PREDS = re.compile(
    r'^(traveled_to|went_to|moved_to|went|traveled|at|is_at|location)\((.+?),(.+?)\)$',
    re.IGNORECASE,
)
_PICKUP_PREDS = re.compile(r'^(picked_up|picked|grabbed|has)\((.+?),(.+?)\)$', re.IGNORECASE)
_DROP_PREDS = re.compile(r'^(dropped|put_down|left)\((.+?),(.+?)\)$', re.IGNORECASE)


def _apply_kinship_rules(atms: BoundedATMS) -> None:
    """Apply kinship transitivity rules to existing L1 nodes."""
    node_ids = list(atms.nodes.keys())
    parents: list[tuple[str, str, str]] = []  # (X, Y, node_id)
    siblings: list[tuple[str, str, str]] = []
    for nid in node_ids:
        m = _PARENT_PREDS.match(nid)
        if m:
            parents.append((m.group(2).strip(), m.group(3).strip(), nid))
            # Also add canonical parent(X,Y) fact if not using that name
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
                # Also fire great-grandparent
                for y3, w, nid3 in parents:
                    if y3 == z:
                        ggp_node = f"great_grandparent({x},{w})"
                        atms._get_or_create(ggp_node)
                        atms.add_rule([nid1, nid2, nid3], ggp_node)

    for x, y, nid in siblings:
        # Symmetry
        sym_node = f"sibling({y},{x})"
        atms._get_or_create(sym_node)
        atms.add_rule([nid], sym_node)

    # parent + sibling → uncle/aunt (generic relation node)
    for px, py, pnid in parents:
        for sx, sy, snid in siblings:
            if py == sx:
                rel_node = f"aunt_or_uncle({px},{sy})"
                atms._get_or_create(rel_node)
                atms.add_rule([pnid, snid], rel_node)


def _apply_location_rules(atms: BoundedATMS) -> None:
    """Apply location rules to existing L1 nodes (bAbI style)."""
    node_ids = list(atms.nodes.keys())
    locations: list[tuple[str, str, str]] = []  # (person, location, node_id)
    pickups: list[tuple[str, str, str]] = []
    drops: list[tuple[str, str, str]] = []
    ats: list[tuple[str, str, str]] = []

    for nid in node_ids:
        m = _TRAVELED_PREDS.match(nid)
        if m:
            p, loc = m.group(2).strip(), m.group(3).strip()
            locations.append((p, loc, nid))
            # traveled_to(X,L) → at(X,L)
            at_node = f"at({p},{loc})"
            atms.add_fact(at_node, env=frozenset())
            ats.append((p, loc, at_node))
        m = _PICKUP_PREDS.match(nid)
        if m:
            person, obj = m.group(2).strip(), m.group(3).strip()
            pickups.append((person, obj, nid))
            # picked_up(X,O) → has(X,O)
            has_node = f"has({person},{obj})"
            atms.add_fact(has_node, env=frozenset())
        m = _DROP_PREDS.match(nid)
        if m:
            drops.append((m.group(2).strip(), m.group(3).strip(), nid))

    # has(X,O) + at(X,L) → at(O,L)
    has_facts = [(p, o, nid) for nid in node_ids if (m := re.match(r'has\((.+?),(.+?)\)', nid)) for p, o in [(m.group(1), m.group(2))]]
    for p1, loc, at_nid in ats:
        for p2, obj, has_nid in has_facts:
            if p1 == p2:
                obj_at_node = f"at({obj},{loc})"
                atms._get_or_create(obj_at_node)
                atms.add_rule([at_nid, has_nid], obj_at_node)

    # same_location: at(X,L) + at(Y,L) → same_location(X,Y)
    for p1, loc1, nid1 in ats:
        for p2, loc2, nid2 in ats:
            if loc1 == loc2 and p1 != p2:
                sl_node = f"same_location({p1},{p2})"
                atms._get_or_create(sl_node)
                atms.add_rule([nid1, nid2], sl_node)


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
    """Answer yes/no if derivable with low assumption load; abstain otherwise."""
    if not score["derivable"]:
        return "abstain"
    # Derivable: if empty-env or low load → yes (conclusion follows), else abstain
    if score["assumption_load"] <= ASSUMPTION_THRESHOLD:
        return "yes"
    # High assumption load: still make a probabilistic guess but mark as uncertain
    return "abstain"


def method_answer_with_fallback(score: dict) -> str:
    """Like method_answer but falls back to 'no' for non-derivable (for accuracy calcs)."""
    ans = method_answer(score)
    # For evaluation purposes: non-derivable → assume "no" (conclusion not supported)
    if ans == "abstain" and not score["derivable"]:
        return "no"
    return ans


# ═══════════════════════════════════════════════════════════════════════════════
# BASELINES
# ═══════════════════════════════════════════════════════════════════════════════

COT_PROMPT = """\
Document: {document}

Question: {query}

Think step by step using ONLY information in the document, then answer.
End your response with exactly "Answer: yes" or "Answer: no" or "Answer: unknown".

Step-by-step reasoning:"""


async def cot_baseline(document: str, query: str) -> tuple[str, float]:
    prompt = COT_PROMPT.format(document=document[:2000], query=query)
    resp, cost = await llm_call(prompt, max_tokens=400)
    answer = parse_yes_no(resp)
    return answer, cost


LINC_PROMPT = """\
Document: {document}

Question: {query}

Reason formally: list the logical steps from document facts to the answer.
Then answer YES or NO based solely on document facts + basic logic.
End with: "Conclusion: yes" or "Conclusion: no" or "Conclusion: unknown"."""


async def linc_sample(document: str, query: str) -> tuple[str, float]:
    prompt = LINC_PROMPT.format(document=document[:2000], query=query)
    resp, cost = await llm_call(prompt, max_tokens=350)
    answer = parse_yes_no(resp, patterns=["conclusion:", "answer:"])
    return answer, cost


async def linc_baseline(document: str, query: str, k: int = LINC_K) -> tuple[str, float]:
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
        return "unknown", total_cost
    counter = Counter(a for a in answers if a != "unknown")
    if counter:
        return counter.most_common(1)[0][0], total_cost
    return "unknown", total_cost


PROBLOG_PROMPT = """\
Document: {document}

Question: {query}

List each explicit fact from the document as: fact(confidence).
Example: parent(alice,bob)(0.99). sibling(bob,carol)(0.99).
Then state if the answer is derivable.
End with: "Derivable: yes" or "Derivable: no"."""


async def problog_baseline(document: str, query: str) -> tuple[str, float, float]:
    """ProbLog approximation: product of weights of required assumptions.
    Returns (answer, cost, derived_prob)."""
    prompt = PROBLOG_PROMPT.format(document=document[:2000], query=query)
    resp, cost = await llm_call(prompt, max_tokens=400)
    answer = parse_yes_no(resp, patterns=["derivable:", "answer:", "conclusion:"])
    # Extract numeric confidence if present
    nums = re.findall(r'\b(0\.\d+|1\.0)\b', resp)
    if nums:
        prob = float(nums[0])
    else:
        prob = 1.0 if answer == "yes" else 0.0
    return answer, cost, prob


def parse_yes_no(
    text: str,
    patterns: list[str] | None = None,
) -> str:
    """Extract yes/no from LLM response text."""
    if patterns is None:
        patterns = ["answer:", "conclusion:", "result:"]
    text_lower = text.lower()
    # Look for pattern + yes/no
    for pat in patterns:
        idx = text_lower.rfind(pat)
        if idx != -1:
            after = text_lower[idx + len(pat):idx + len(pat) + 30]
            if "yes" in after:
                return "yes"
            if "no" in after:
                return "no"
    # Fallback: last occurrence of yes/no
    last_yes = text_lower.rfind("yes")
    last_no = text_lower.rfind("no")
    if last_yes == -1 and last_no == -1:
        return "unknown"
    if last_yes > last_no:
        return "yes"
    return "no"


# ═══════════════════════════════════════════════════════════════════════════════
# DATASETS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Instance:
    doc_id: str
    document: str
    query: str
    gold_answer: str  # "yes" or "no"
    hop_count: int
    source: str  # "clutrr", "babi", "custom"
    query_node: str  # ATMS node ID for the query conclusion


def load_clutrr(max_per_bucket: int = 20) -> list[Instance]:
    """Load CLUTRR kinship reasoning dataset from HuggingFace."""
    logger.info("Loading CLUTRR dataset...")
    try:
        from datasets import load_dataset
        # Try multiple config options
        ds = None
        for name in ["CLUTRR/clutrr", "clutrr"]:
            for split_cfg in [
                ("gen_train233", None),
                ("gen_train", None),
                (None, "train"),
            ]:
                try:
                    if split_cfg[0]:
                        ds = load_dataset(name, split_cfg[0], trust_remote_code=True)
                        # It's a DatasetDict — pick the first split
                        if hasattr(ds, "keys"):
                            split_name = list(ds.keys())[0]
                            ds = ds[split_name]
                    else:
                        ds = load_dataset(name, split=split_cfg[1], trust_remote_code=True)
                    logger.info(f"Loaded CLUTRR from {name} with config {split_cfg}")
                    break
                except Exception as e:
                    logger.debug(f"CLUTRR load attempt failed ({name},{split_cfg}): {e}")
                    continue
            if ds is not None:
                break

        if ds is None:
            logger.warning("CLUTRR unavailable, using synthetic fallback")
            return _synthetic_clutrr()

        instances = []
        hop_buckets: dict[int, list] = {}
        for row in ds:
            story = str(row.get("story", row.get("passage", row.get("input_text", ""))))
            query_text = str(row.get("query", row.get("question", "")))
            answer = str(row.get("answer", row.get("target", ""))).lower().strip()
            n_hops = int(row.get("n_supporting_facts", row.get("num_hops", row.get("k", 2))))

            if not story or not query_text or len(story) > 4000:
                continue
            # Normalize answer to yes/no
            gold = "yes" if answer in ("yes", "true", "1") else "no"
            hop_buckets.setdefault(n_hops, []).append((story, query_text, gold, n_hops))

        for hop, bucket in sorted(hop_buckets.items()):
            for story, query_text, gold, n_hops in bucket[:max_per_bucket]:
                doc_id = f"clutrr_{hop}_{len(instances)}"
                # Build query node: try to extract relationship from query
                qnode = _build_query_node(query_text, story)
                instances.append(Instance(
                    doc_id=doc_id,
                    document=story,
                    query=query_text,
                    gold_answer=gold,
                    hop_count=n_hops,
                    source="clutrr",
                    query_node=qnode,
                ))
        logger.info(f"CLUTRR: {len(instances)} instances across {len(hop_buckets)} hop buckets")
        return instances if instances else _synthetic_clutrr()

    except Exception:
        logger.exception("CLUTRR loading failed")
        return _synthetic_clutrr()


def _build_query_node(query: str, story: str) -> str:
    """Build a canonical ATMS node ID for a query about a relationship."""
    # Look for "[Name] is [Name]'s [relation]?" patterns
    patterns = [
        r'([\w]+)\s+is\s+([\w]+)[\'’]?s\s+([\w_]+)',
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
    # Fallback: use query hash as node
    return f"query_{abs(hash(query)) % 10000}"


def load_babi(n_task15: int = 50, n_task16: int = 50) -> list[Instance]:
    """Load bAbI tasks 15 (basic deduction) and 16 from HuggingFace."""
    logger.info("Loading bAbI dataset...")
    try:
        from datasets import load_dataset
        ds = None
        for name in ["facebook/babi", "Maaz/babi_tasks_1-20_v1.2", "metaeval/babi_nli"]:
            try:
                ds = load_dataset(name, trust_remote_code=True)
                logger.info(f"Loaded bAbI from {name}")
                break
            except Exception as e:
                logger.debug(f"bAbI load attempt failed ({name}): {e}")

        if ds is None:
            logger.warning("bAbI unavailable, using synthetic fallback")
            return _synthetic_babi()

        instances = []
        for split_name in ds.keys():
            split = ds[split_name]
            for row in split:
                task_id = int(row.get("task_id", row.get("task", 0)))
                if task_id not in (15, 16):
                    continue
                story = str(row.get("story", row.get("passage", row.get("context", ""))))
                query_text = str(row.get("question", row.get("query", "")))
                answer = str(row.get("answer", "")).lower().strip()

                if not story or not query_text:
                    continue
                gold = "yes" if answer in ("yes", "true") else "no"
                src_task = "babi15" if task_id == 15 else "babi16"
                limit = n_task15 if task_id == 15 else n_task16
                if sum(1 for x in instances if x.source == src_task) >= limit:
                    continue

                qnode = f"babi_q_{abs(hash(query_text)) % 10000}"
                instances.append(Instance(
                    doc_id=f"babi_{task_id}_{len(instances)}",
                    document=story,
                    query=query_text,
                    gold_answer=gold,
                    hop_count=task_id - 14,
                    source="babi",
                    query_node=qnode,
                ))

        logger.info(f"bAbI: {len(instances)} instances")
        return instances if instances else _synthetic_babi()

    except Exception:
        logger.exception("bAbI loading failed")
        return _synthetic_babi()


def _synthetic_clutrr() -> list[Instance]:
    """Synthetic kinship stories for when CLUTRR is unavailable."""
    templates = [
        # (story, query, gold, hops, query_node)
        (
            "Alice is Bob's mother. Bob is Carol's father.",
            "Is Alice Carol's grandmother?",
            "yes", 2,
            "grandparent(alice,carol)",
        ),
        (
            "John married Mary. Mary is Lisa's mother. John is Peter's father.",
            "Are Lisa and Peter siblings?",
            "yes", 2,
            "sibling(lisa,peter)",
        ),
        (
            "Tom is Jane's father. Jane is Ann's mother. Ann is Sue's mother.",
            "Is Tom Sue's great-grandfather?",
            "yes", 3,
            "great_grandparent(tom,sue)",
        ),
        (
            "Emily is David's sister. David is Frank's son.",
            "Is Emily Frank's daughter?",
            "yes", 2,
            "parent(frank,emily)",
        ),
        (
            "Sam is Kate's uncle. Kate is Lily's mother.",
            "Is Sam Lily's great-uncle?",
            "yes", 2,
            "relation(sam,lily)",
        ),
        (
            "Paul is Sarah's grandfather. Sarah has a brother named Jake.",
            "Is Paul Jake's grandfather?",
            "yes", 2,
            "grandparent(paul,jake)",
        ),
        (
            "Maria is Carlos's mother. Carlos is Ana's husband.",
            "Is Maria Ana's mother-in-law?",
            "yes", 2,
            "relation(maria,ana)",
        ),
        (
            "Helen is George's mother. George has a son named Tim.",
            "Is Helen Tim's grandmother?",
            "yes", 2,
            "grandparent(helen,tim)",
        ),
        (
            "Lucy is Mark's daughter. Mark is Susan's brother.",
            "Is Susan Lucy's aunt?",
            "yes", 2,
            "relation(susan,lucy)",
        ),
        (
            "Bob is Alice's father. Alice is Charlie's mother. Charlie is Dana's father.",
            "Is Bob Dana's great-grandfather?",
            "yes", 3,
            "great_grandparent(bob,dana)",
        ),
    ]
    instances = []
    for i, (story, query, gold, hops, qnode) in enumerate(templates):
        instances.append(Instance(
            doc_id=f"synth_clutrr_{i}",
            document=story,
            query=query,
            gold_answer=gold,
            hop_count=hops,
            source="clutrr",
            query_node=qnode,
        ))
    logger.info(f"Using {len(instances)} synthetic CLUTRR instances")
    return instances


def _synthetic_babi() -> list[Instance]:
    """Synthetic bAbI-style location stories."""
    templates = [
        (
            "Mary traveled to the office. John went to the kitchen. Mary went to the garden.",
            "Is Mary in the garden?",
            "yes", 1, "at(mary,garden)",
        ),
        (
            "Daniel went to the bedroom. Sandra traveled to the office. Daniel moved to the kitchen.",
            "Is Daniel in the kitchen?",
            "yes", 1, "at(daniel,kitchen)",
        ),
        (
            "John picked up the football. John went to the bedroom. John dropped the football.",
            "Is the football in the bedroom?",
            "yes", 2, "at(football,bedroom)",
        ),
        (
            "Sandra went to the garden. Mary traveled to the office. Sandra moved to the kitchen.",
            "Is Sandra in the office?",
            "no", 1, "at(sandra,office)",
        ),
        (
            "Daniel picked up the milk. Daniel traveled to the garden. Sandra went to the bedroom.",
            "Does Daniel have the milk?",
            "yes", 1, "has(daniel,milk)",
        ),
        (
            "Mary went to the office. Mary picked up the apple. Mary traveled to the garden.",
            "Is the apple in the garden?",
            "yes", 2, "at(apple,garden)",
        ),
        (
            "John went to the bedroom. Sandra went to the bedroom.",
            "Are John and Sandra in the same place?",
            "yes", 1, "same_location(john,sandra)",
        ),
        (
            "Daniel traveled to the office. Daniel picked up the football. Daniel moved to the kitchen. Daniel dropped the football.",
            "Is the football in the kitchen?",
            "yes", 3, "at(football,kitchen)",
        ),
    ]
    instances = []
    for i, (story, query, gold, hops, qnode) in enumerate(templates):
        instances.append(Instance(
            doc_id=f"synth_babi_{i}",
            document=story,
            query=query,
            gold_answer=gold,
            hop_count=hops,
            source="babi",
            query_node=qnode,
        ))
    logger.info(f"Using {len(instances)} synthetic bAbI instances")
    return instances


CUSTOM_DOCS = [
    {
        "document": "In Smith v. Jones (2021), the court ruled that Smith, who is the landlord of the disputed property, had failed to provide adequate notice before eviction. The tenant, Jones, had lived at the property for five years. The judge, Honorable Williams, ordered compensation of $5,000.",
        "queries": [
            ("Does Smith own the disputed property?", "yes", 1, "owns(smith,disputed_property)"),
            ("Was Jones evicted without proper notice?", "yes", 2, "evicted_improperly(jones)"),
            ("Did the judge award Jones money?", "yes", 2, "awarded(jones,compensation)"),
        ],
    },
    {
        "document": "The company TechCorp announced that CEO Alice Brown would resign effective January 2024. The board appointed CFO Robert Green as interim CEO. Employees were notified by email.",
        "queries": [
            ("Is Alice Brown leaving TechCorp?", "yes", 1, "leaving(alice_brown,techcorp)"),
            ("Is Robert Green currently the CEO?", "yes", 2, "is_ceo(robert_green,techcorp)"),
            ("Were employees informed of the change?", "yes", 1, "informed(employees,change)"),
        ],
    },
    {
        "document": "Goldilocks entered the house of the three bears. She found three bowls of porridge. She tasted Papa Bear's bowl first but it was too hot. Mama Bear's bowl was too cold. Baby Bear's bowl was just right, so she ate it all.",
        "queries": [
            ("Did Goldilocks eat Baby Bear's porridge?", "yes", 2, "ate(goldilocks,baby_bear_porridge)"),
            ("Was Papa Bear's porridge too cold?", "no", 1, "too_cold(papa_bear_porridge)"),
            ("Did Goldilocks find the house empty?", "no", 1, "empty_house(bears_house)"),
        ],
    },
    {
        "document": "The contract between Acme Corp and Widget Ltd stipulates that Widget Ltd must deliver 500 units per month. Acme Corp agreed to pay $10 per unit. Widget Ltd delivered only 300 units in March 2023, citing supply chain issues.",
        "queries": [
            ("Did Widget Ltd fulfill its contractual delivery obligation in March 2023?", "no", 2, "fulfilled_obligation(widget_ltd,march_2023)"),
            ("Is the payment rate $10 per unit?", "yes", 1, "payment_rate(10,per_unit)"),
            ("Did Widget Ltd cite a reason for the shortfall?", "yes", 1, "cited_reason(widget_ltd,shortfall)"),
        ],
    },
    {
        "document": "Jack and Jill went up the hill to fetch a pail of water. Jack fell down and broke his crown, and Jill came tumbling after. The accident happened on a steep slope near the village well.",
        "queries": [
            ("Did Jack get injured?", "yes", 1, "injured(jack)"),
            ("Were both Jack and Jill hurt?", "yes", 2, "both_hurt(jack,jill)"),
            ("Were they fetching water from a river?", "no", 1, "water_source(river)"),
        ],
    },
    {
        "document": "Mayor Thompson signed the new zoning ordinance on March 15. The ordinance restricts building heights in the downtown area to 6 stories. Developer Reynolds had already submitted plans for a 10-story building.",
        "queries": [
            ("Is the 10-story building plan compliant with the new ordinance?", "no", 2, "compliant(reynolds_plan,ordinance)"),
            ("Did the mayor approve the zoning change?", "yes", 1, "approved(thompson,zoning_ordinance)"),
            ("Are buildings downtown now limited in height?", "yes", 1, "height_restricted(downtown)"),
        ],
    },
    {
        "document": "Dr. Chen is the head of the cardiology department at City Hospital. The hospital is accredited by the National Medical Board. Dr. Chen published a study on heart failure prevention last year.",
        "queries": [
            ("Does Dr. Chen work at an accredited hospital?", "yes", 2, "works_at_accredited(dr_chen)"),
            ("Is Dr. Chen a cardiologist?", "yes", 1, "is_cardiologist(dr_chen)"),
            ("Did Dr. Chen recently publish research?", "yes", 1, "published_research(dr_chen)"),
        ],
    },
    {
        "document": "The treaty was signed between Nation A and Nation B in 2018. Under the treaty, Nation A agreed to reduce tariffs on Nation B's agricultural products. Nation B committed to open its ports to Nation A's shipping vessels. In 2022, Nation A imposed new tariffs citing security concerns.",
        "queries": [
            ("Did Nation A violate the 2018 treaty?", "yes", 2, "violated_treaty(nation_a)"),
            ("Were tariffs reduced when the treaty was signed?", "yes", 1, "reduced_tariffs(nation_a,2018)"),
            ("Did Nation B gain port access rights?", "yes", 1, "port_access(nation_b)"),
        ],
    },
    {
        "document": "The Red Riding Hood went to visit her grandmother who lived in the forest. Her mother told her to stay on the path. She met a wolf who asked where she was going. She told the wolf her grandmother's address.",
        "queries": [
            ("Did Red Riding Hood reveal her grandmother's location?", "yes", 2, "revealed_location(red_riding_hood,grandmother)"),
            ("Did Red Riding Hood's mother warn her?", "yes", 1, "warned(mother,red_riding_hood)"),
            ("Did Red Riding Hood go to a city?", "no", 1, "destination(red_riding_hood,city)"),
        ],
    },
    {
        "document": "Plaintiff Martinez filed suit against Defendant Corp for breach of contract in June 2023. The contract specified a delivery date of March 1, 2023. Defendant Corp delivered the goods on April 15, 2023. The plaintiff claims damages of $50,000.",
        "queries": [
            ("Was the delivery late?", "yes", 1, "late_delivery(defendant_corp)"),
            ("Is Martinez seeking compensation?", "yes", 1, "seeking_damages(martinez)"),
            ("Was the contract for services?", "no", 1, "contract_type(services)"),
        ],
    },
]


def load_custom_docs() -> list[Instance]:
    """Load custom document set with multi-hop queries."""
    instances = []
    for i, doc_data in enumerate(CUSTOM_DOCS):
        document = doc_data["document"]
        for j, (query, gold, hops, qnode) in enumerate(doc_data["queries"]):
            instances.append(Instance(
                doc_id=f"custom_{i}_{j}",
                document=document,
                query=query,
                gold_answer=gold,
                hop_count=hops,
                source="custom",
                query_node=qnode,
            ))
    logger.info(f"Custom docs: {len(instances)} instances")
    return instances


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 0: TRACTABILITY PILOT
# ═══════════════════════════════════════════════════════════════════════════════

async def run_single_instance(inst: Instance) -> dict:
    """Full pipeline for one instance. Returns result dict."""
    t0 = time.perf_counter()

    # L1 grounding
    l1_atoms, l1_cost = await extract_l1_atoms(inst.document)

    # L2 assumptions
    l2, l2_cost = await propose_l2_assumptions(l1_atoms, inst.document, inst.query)

    # ATMS
    domain = "babi" if "babi" in inst.source else "clutrr"
    atms = build_atms(l1_atoms, l2, inst.query_node, domain=domain)

    # Score
    score = score_conclusion(atms, inst.query_node)
    our_ans = method_answer(score)
    # Fallback: non-derivable → "no" (conclusion not supported by document)
    our_ans_eval = method_answer_with_fallback(score)

    n_choice_points = len(l2.get("unifications", [])) + len(l2.get("bridge_rules", []))
    wall_clock = time.perf_counter() - t0

    return {
        "doc_id": inst.doc_id,
        "source": inst.source,
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
        "our_method_eval": our_ans_eval,  # for accuracy: abstain→no
        "l1_cost": l1_cost,
        "l2_cost": l2_cost,
        "total_cost_this": l1_cost + l2_cost,
    }


async def run_baselines(inst: Instance) -> dict:
    """Run all three baselines on one instance."""
    cot_ans, cot_cost = await cot_baseline(inst.document, inst.query)
    linc_ans, linc_cost = await linc_baseline(inst.document, inst.query, k=LINC_K)
    problog_ans, pb_cost, pb_prob = await problog_baseline(inst.document, inst.query)
    return {
        "cot": cot_ans,
        "linc": linc_ans,
        "problog": problog_ans,
        "problog_prob": pb_prob,
        "cot_cost": cot_cost,
        "linc_cost": linc_cost,
        "problog_cost": pb_cost,
    }


async def stage0_pilot(pilot_instances: list[Instance]) -> dict:
    """Stage 0 tractability pilot. Returns metrics and decision."""
    logger.info(f"Stage 0: running tractability pilot on {len(pilot_instances)} docs")
    tasks = [run_single_instance(inst) for inst in pilot_instances]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    valid = [r for r in results if isinstance(r, dict)]
    if not valid:
        return {"decision": "tractability_failure", "mean_choice_points": 0, "mean_wall_clock_sec": 999}

    mean_cp = float(np.mean([r["n_choice_points"] for r in valid]))
    mean_wc = float(np.mean([r["wall_clock_sec"] for r in valid]))
    mean_ng = float(np.mean([r["n_nogoods"] for r in valid]))
    mean_cost = float(np.mean([r["total_cost_this"] for r in valid]))

    logger.info(f"Stage 0: mean_choice_points={mean_cp:.1f} mean_wall_clock={mean_wc:.2f}s mean_cost=${mean_cost:.5f}")

    if mean_wc < 30.0 and mean_cp <= 20:
        decision = "exact_atms"
    elif mean_wc < 90.0:
        decision = "exact_atms_reduced"
        global DEPTH_CAP, BEAM_WIDTH
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
# EVALUATION METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_metrics(results: list[dict]) -> dict:
    """Compute all evaluation metrics from full results."""
    # Filter usable results (our method produced an answer)
    answered = [r for r in results if r["our_method"] != "abstain"]
    all_results = results

    # H1: Empty-environment precision (use eval answer: abstain→no)
    ee_results = [r for r in all_results if r["empty_env_derivable"]]
    ee_precision = (
        float(np.mean([r.get("our_method_eval", r["our_method"]) == r["gold"] for r in ee_results]))
        if ee_results else float("nan")
    )

    # L1 grounding error rate
    total_atoms = sum(len(r["l1_atoms"]) for r in results)
    grounding_errors = sum(
        1 for r in results for a in r["l1_atoms"]
        if not a.get("grounding_verified", True)
    )
    l1_error_rate = grounding_errors / max(total_atoms, 1)

    # H2: Spearman correlation of assumption_load vs error
    finite_results = [
        r for r in all_results
        if r["assumption_load"] < float("inf") and r["assumption_load"] != float("inf")
    ]
    spearman_rho = float("nan")
    spearman_p = float("nan")
    auroc_load = float("nan")

    if len(finite_results) >= 10:
        loads = [r["assumption_load"] for r in finite_results]
        errors = [1 if r.get("our_method_eval", r["our_method"]) != r["gold"] else 0 for r in finite_results]
        try:
            rho, p = spearmanr(loads, errors)
            spearman_rho = float(rho)
            spearman_p = float(p)
        except Exception:
            pass
        # AUROC
        if len(set(errors)) > 1:
            try:
                auroc_load = float(roc_auc_score(errors, loads))
            except Exception:
                pass

    # LLM confidence AUROC (use problog_prob as proxy)
    auroc_llm_conf = float("nan")
    if len(finite_results) >= 10:
        errors2 = [1 if r.get("our_method_eval", r["our_method"]) != r["gold"] else 0 for r in finite_results]
        probs = [1 - r.get("problog_prob", 0.5) for r in finite_results]  # higher prob = less error
        if len(set(errors2)) > 1:
            try:
                auroc_llm_conf = float(roc_auc_score(errors2, probs))
            except Exception:
                pass

    # H3: Risk-coverage AUC
    def risk_coverage_auc(results_list: list[dict], score_key: str, answer_key: str, gold_key: str = "gold") -> float:
        """Lower AUC = better (less risk for given coverage)."""
        scores = [r.get(score_key, float("inf")) for r in results_list]
        finite_scores = [s for s in scores if s < float("inf")]
        if not finite_scores:
            return float("nan")
        thresholds = sorted(set(finite_scores))
        areas = []
        prev_cov = 0.0
        for t in thresholds:
            answered_here = [r for r, s in zip(results_list, scores) if s <= t]
            if not answered_here:
                continue
            coverage = len(answered_here) / len(results_list)
            risk = float(np.mean([r[answer_key] != r[gold_key] for r in answered_here]))
            areas.append((coverage - prev_cov) * risk)
            prev_cov = coverage
        return float(sum(areas)) if areas else float("nan")

    our_rc_auc = risk_coverage_auc(all_results, "assumption_load", "our_method_eval")
    cot_rc_auc = _method_rc_auc(all_results, "cot")
    linc_rc_auc = _method_rc_auc(all_results, "linc")
    problog_rc_auc = _method_rc_auc(all_results, "problog")

    # H4: implicit vs explicit advantage (our method vs CoT)
    implicit = [r for r in all_results if r["source"] in ("clutrr", "babi", "custom")]
    explicit = [r for r in all_results if r["source"] not in ("clutrr", "babi", "custom")]

    def acc(results_subset, method_key):
        if not results_subset:
            return float("nan")
        key = method_key if method_key != "our_method" else "our_method_eval"
        return float(np.mean([r.get(key, r.get(method_key, "unknown")) == r["gold"] for r in results_subset]))

    def answered_acc(results_subset, method_key):
        """Accuracy on non-abstain items only."""
        answered = [r for r in results_subset if r.get(method_key, "abstain") != "abstain"]
        if not answered:
            return float("nan")
        return float(np.mean([r[method_key] == r["gold"] for r in answered]))

    abstain_rate = float(np.mean([r["our_method"] == "abstain" for r in all_results]))
    answered_accuracy_our = answered_acc(all_results, "our_method")

    implicit_our_acc = acc(implicit, "our_method")
    implicit_cot_acc = acc(implicit, "cot")
    implicit_adv = implicit_our_acc - implicit_cot_acc if not (math.isnan(implicit_our_acc) or math.isnan(implicit_cot_acc)) else float("nan")

    explicit_our_acc = acc(explicit, "our_method")
    explicit_cot_acc = acc(explicit, "cot")
    explicit_adv = explicit_our_acc - explicit_cot_acc if not (math.isnan(explicit_our_acc) or math.isnan(explicit_cot_acc)) else float("nan")

    # ATMS vs ProbLog divergence
    problog_div = [
        r for r in all_results
        if r.get("problog") != r["our_method_eval"] and r["empty_env_derivable"]
    ]

    return {
        "l1_grounding_error_rate": l1_error_rate,
        "empty_env_precision": ee_precision,
        "n_empty_env_derivable": len(ee_results),
        "spearman_rho": spearman_rho,
        "spearman_p": spearman_p,
        "auroc_assumption_load": auroc_load,
        "auroc_llm_confidence": auroc_llm_conf,
        "risk_coverage_auc": {
            "our_method": our_rc_auc,
            "cot": cot_rc_auc,
            "linc_vote": linc_rc_auc,
            "problog": problog_rc_auc,
        },
        "implicit_advantage_gap": implicit_adv,
        "explicit_advantage_gap": explicit_adv,
        "abstain_rate": abstain_rate,
        "answered_accuracy_our_method": answered_accuracy_our,
        "overall_accuracy": {
            "our_method": acc(all_results, "our_method"),
            "cot": acc(all_results, "cot"),
            "linc": acc(all_results, "linc"),
            "problog": acc(all_results, "problog"),
        },
        "problog_atms_divergence_rate": len(problog_div) / max(len(all_results), 1),
    }


def _method_rc_auc(results: list[dict], method_key: str) -> float:
    """Risk-coverage AUC for a binary method (yes/no/unknown → answered or not)."""
    answered = [(i, r) for i, r in enumerate(results) if r.get(method_key, "unknown") != "unknown"]
    if not answered:
        return float("nan")
    n = len(results)
    # For baseline: coverage = fraction answered; risk = error rate among answered
    coverage = len(answered) / n
    risk = float(np.mean([r[method_key] != r["gold"] for _, r in answered]))
    return risk * coverage  # simple single-threshold AUC approximation


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN EXPERIMENT
# ═══════════════════════════════════════════════════════════════════════════════

async def process_instance_full(inst: Instance) -> dict | None:
    """Run our method + all baselines on one instance."""
    global _cumulative_cost
    if _cumulative_cost >= COST_CAP:
        logger.warning(f"Cost cap reached at ${_cumulative_cost:.4f}, stopping")
        return None
    try:
        # Run our method and baselines concurrently
        method_task = asyncio.create_task(run_single_instance(inst))
        baseline_task = asyncio.create_task(run_baselines(inst))
        method_result, baseline_result = await asyncio.gather(method_task, baseline_task)

        result = {**method_result, **baseline_result}
        result["cumulative_cost"] = _cumulative_cost
        return result
    except Exception:
        logger.exception(f"Failed on {inst.doc_id}")
        return None


async def run_experiment(instances: list[Instance], max_concurrent: int = 6) -> list[dict]:
    """Run full experiment with controlled concurrency."""
    results = []
    batch_size = max_concurrent

    logger.info(f"Running full experiment on {len(instances)} instances (batch={batch_size})")
    for i in range(0, len(instances), batch_size):
        if _cumulative_cost >= COST_CAP:
            logger.warning("Cost cap reached, stopping experiment")
            break

        batch = instances[i : i + batch_size]
        tasks = [process_instance_full(inst) for inst in batch]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

        for r in batch_results:
            if isinstance(r, dict):
                results.append(r)
            elif isinstance(r, Exception):
                logger.warning(f"Batch item failed: {r}")

        done = len(results)
        logger.info(f"Progress: {done}/{len(instances)} | cost=${_cumulative_cost:.4f}")
        gc.collect()

    return results


def format_result_for_schema(results: list[dict]) -> dict:
    """Format results into exp_gen_sol_out schema."""
    # Group by source dataset
    by_source: dict[str, list[dict]] = {}
    for r in results:
        src = r["source"]
        by_source.setdefault(src, []).append(r)

    datasets = []
    for src, src_results in by_source.items():
        examples = []
        for r in src_results:
            # Build input string
            input_str = f"Document: {r.get('document', r.get('query', ''))[:500]}\n\nQuery: {r['query']}"
            # Build output string (gold answer)
            output_str = r["gold"]

            # Safe serialization helper
            def safe_str(val: Any) -> str:
                if val is None or (isinstance(val, float) and math.isnan(val)):
                    return "null"
                if isinstance(val, float) and math.isinf(val):
                    return "inf"
                return str(val)

            # Assumption load as string (may be inf or float)
            al = r.get("assumption_load", float("inf"))
            al_str = "inf" if al == float("inf") else str(al)

            example = {
                "input": input_str,
                "output": output_str,
                "predict_our_method": r.get("our_method_eval", r.get("our_method", "unknown")),
                "predict_cot": r.get("cot", "unknown"),
                "predict_linc": r.get("linc", "unknown"),
                "predict_problog": r.get("problog", "unknown"),
                "metadata_hop_count": str(r.get("hop_count", 0)),
                "metadata_assumption_load": al_str,
                "metadata_empty_env_derivable": str(r.get("empty_env_derivable", False)),
                "metadata_n_choice_points": str(r.get("n_choice_points", 0)),
                "metadata_cumulative_cost_usd": str(round(r.get("cumulative_cost", 0), 5)),
                "metadata_wall_clock_sec": str(round(r.get("wall_clock_sec", 0), 3)),
                "metadata_l1_grounding_verified": str(
                    all(a.get("grounding_verified", True) for a in r.get("l1_atoms", []))
                ),
            }
            examples.append(example)
        datasets.append({"dataset": src, "examples": examples})

    return {"datasets": datasets}


@logger.catch(reraise=True)
async def main() -> None:
    logger.info("=== ATMS Hallucination Meter Experiment ===")
    logger.info(f"Model: {CHEAP_MODEL} | Cost cap: ${COST_CAP}")

    if not OPENROUTER_KEY:
        logger.error("OPENROUTER_API_KEY not set")
        sys.exit(1)

    # Load datasets
    logger.info("Loading datasets...")
    clutrr = load_clutrr(max_per_bucket=15)
    babi = load_babi(n_task15=30, n_task16=20)
    custom = load_custom_docs()

    all_instances = clutrr + babi + custom
    logger.info(f"Total instances: {len(all_instances)} (CLUTRR={len(clutrr)}, bAbI={len(babi)}, custom={len(custom)})")

    # Store document text on results for formatting
    inst_map = {inst.doc_id: inst for inst in all_instances}

    # Stage 0: tractability pilot (first 10 clutrr + 5 babi)
    pilot = clutrr[:10] + babi[:5]
    stage0 = await stage0_pilot(pilot)
    logger.info(f"Stage 0 decision: {stage0['decision']}")

    if stage0["decision"] == "tractability_failure":
        logger.error("Tractability failure — writing partial results")
        out = {
            "metadata": {"stage0_tractability": stage0, "error": "tractability_failure"},
            "datasets": [],
        }
        (RESULTS_DIR / "method_out.json").write_text(json.dumps(out, indent=2))
        return

    # Full experiment
    results = await run_experiment(all_instances, max_concurrent=6)
    logger.info(f"Completed {len(results)} instances | total_cost=${_cumulative_cost:.4f}")

    # Attach document text for formatting
    for r in results:
        doc_id = r.get("doc_id", "")
        inst = inst_map.get(doc_id)
        if inst:
            r["document"] = inst.document

    # Compute metrics
    metrics = compute_metrics(results)
    logger.info(f"Metrics: spearman_rho={metrics['spearman_rho']:.3f} ee_precision={metrics['empty_env_precision']:.3f}")
    logger.info(f"Accuracy: our={metrics['overall_accuracy']['our_method']:.3f} cot={metrics['overall_accuracy']['cot']:.3f} abstain_rate={metrics['abstain_rate']:.3f}")

    # Build output JSON (schema format)
    schema_out = format_result_for_schema(results)
    schema_out["metadata"] = {
        "method_name": "Bounded-ATMS Hallucination Meter",
        "description": "Two-layer neuro-symbolic pipeline with L1 grounding + L2 ATMS assumption tracking",
        "parameters": {
            "depth_cap": DEPTH_CAP,
            "beam_width": BEAM_WIDTH,
            "assumption_threshold": ASSUMPTION_THRESHOLD,
            "linc_k": LINC_K,
            "model": CHEAP_MODEL,
        },
        "stage0_tractability": stage0,
        "evaluation": {
            "l1_grounding_error_rate": metrics["l1_grounding_error_rate"],
            "empty_env_precision": metrics["empty_env_precision"],
            "n_empty_env_derivable": metrics["n_empty_env_derivable"],
            "spearman_rho": metrics["spearman_rho"],
            "spearman_p": metrics["spearman_p"],
            "auroc_assumption_load": metrics["auroc_assumption_load"],
            "auroc_llm_confidence": metrics["auroc_llm_confidence"],
            "risk_coverage_auc": metrics["risk_coverage_auc"],
            "implicit_advantage_gap": metrics["implicit_advantage_gap"],
            "explicit_advantage_gap": metrics["explicit_advantage_gap"],
            "overall_accuracy": metrics["overall_accuracy"],
            "abstain_rate": metrics["abstain_rate"],
            "answered_accuracy_our_method": metrics["answered_accuracy_our_method"],
            "problog_atms_divergence_rate": metrics["problog_atms_divergence_rate"],
        },
        "total_cost_usd": _cumulative_cost,
        "n_instances": len(results),
    }

    # Save
    out_path = RESULTS_DIR / "method_out.json"
    out_path.write_text(json.dumps(schema_out, indent=2, default=str))
    logger.info(f"Saved results to {out_path}")

    # Check file size and split if needed
    size_mb = out_path.stat().st_size / 1024 / 1024
    logger.info(f"Output file size: {size_mb:.2f} MB")

    # Close session
    global _session
    if _session and not _session.closed:
        await _session.close()

    logger.info("=== Experiment complete ===")
    logger.info(f"Total cost: ${_cumulative_cost:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
