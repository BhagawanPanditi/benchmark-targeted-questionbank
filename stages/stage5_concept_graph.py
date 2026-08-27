"""Stage 5 — Prerequisite concept graph (done ONCE per domain).

For each concept in the normalized vocabulary (including new concepts added by
Stage 4), one LLM call returns its DIRECT prerequisites (max 5). New
prerequisites missing from the vocabulary are registered in the taxonomy file.

After all concepts are processed:
  * cycles are detected with DFS and broken by removing the edge whose SOURCE
    concept has the lower centrality (degree) in the graph; all removed edges
    are logged and recorded in "removed_cycles"
  * the transitive prerequisite closure (up to depth 3) is computed per concept

Resumability (two layers):
  * per-concept results are checkpointed to
    ``<domain>_concept_graph.json.progress.json`` after every concept, so a
    crashed run resumes where it stopped. The progress file is deleted once the
    final graph is written.
  * a completed graph records which concepts it processed (``"processed"``).
    On a clean re-run, per-concept results are re-derived from the existing
    graph's edges, so the stage makes zero LLM calls ("done ONCE per domain").
    Concepts newly added to the taxonomy afterwards are the only ones re-called.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from string import Template
from typing import Any

from tqdm import tqdm

from utils.io import load_json_obj, require_file, save_json
from utils.llm import LLMError, call_llm

logger = logging.getLogger(__name__)

MAX_PREREQUISITES = 5
MAX_TRANSITIVE_DEPTH = 3

PROMPT_PREREQ = Template(r"""You are a ${domain} curriculum expert building a prerequisite dependency graph.

Target concept: "${concept}"

What concepts must a learner ALREADY understand before they can properly learn
or apply "${concept}"?

Rules:
  - List only DIRECT prerequisites — concepts one step back in the dependency chain.
    Do NOT list transitive prerequisites (those will be found by traversing the graph).
  - Prefer concepts from this known vocabulary (use exact strings where possible):
    ${all_concepts}
  - If a genuine direct prerequisite is missing from the vocabulary entirely, add it
    in the same dot-notation format and mark it new.
  - If this concept is ATOMIC — meaning it has no meaningful prerequisites in a
    ${domain} learning context, it is a true starting point — return an empty list
    and set is_leaf=true.
  - Maximum 5 prerequisites.
  - Be conservative: only list concepts that are genuinely REQUIRED to understand
    "${concept}", not merely helpful or related.

Examples of correct prerequisite relationships:
  algorithms.technique.binary-search requires:
    → algorithms.sorting.sorted-order-property
    → algorithms.iteration.loop-invariants
  dp.technique.memoization requires:
    → algorithms.recursion.recursive-decomposition
    → data-structures.mapping.dictionary
  graph.traversal.dijkstra requires:
    → graph.representation.adjacency-list
    → data-structures.heap.min-heap
    → graph.traversal.bfs (as conceptual foundation)

Return ONLY valid JSON:
{
  "concept": "${concept}",
  "is_leaf": true | false,
  "prerequisites": [
    {"name": "exact.concept.string", "is_new": false},
    ...
  ]
}""")


def _coerce_result(concept: str, data: Any) -> dict:
    """Normalize one LLM prerequisite response (cap at MAX_PREREQUISITES)."""
    result: dict[str, Any] = {"concept": concept, "is_leaf": False, "prerequisites": []}
    if not isinstance(data, dict):
        result["failed"] = True
        return result
    result["is_leaf"] = bool(data.get("is_leaf", False))
    seen: set[str] = set()
    for entry in data.get("prerequisites", []):
        if len(result["prerequisites"]) >= MAX_PREREQUISITES:
            break
        if isinstance(entry, dict):
            name = str(entry.get("name", "")).strip()
            is_new = bool(entry.get("is_new", False))
        elif isinstance(entry, str):
            name = entry.strip()
            is_new = False
        else:
            continue
        name = name.lower()
        if not name or name == concept or name in seen:
            continue
        seen.add(name)
        result["prerequisites"].append({"name": name, "is_new": is_new})
    if not result["prerequisites"]:
        result["is_leaf"] = True
    return result


def _find_back_edge(
    nodes: list[str], adj: dict[str, list[str]]
) -> tuple[str, str, list[str]] | None:
    """Iterative DFS. Returns (u, v, cycle_path) for a back edge u->v, else None.

    ``cycle_path`` is the on-stack path from v (inclusive) to u (inclusive), so
    the cycle's edges are the consecutive pairs of cycle_path plus (u, v).
    """
    WHITE, GRAY, BLACK = 0, 1, 2
    color: dict[str, int] = {n: WHITE for n in nodes}
    for root in nodes:
        if color[root] != WHITE:
            continue
        color[root] = GRAY
        path = [root]
        stack: list[tuple[str, Any]] = [(root, iter(sorted(adj.get(root, []))))]
        while stack:
            node, neighbors = stack[-1]
            descended = False
            for nxt in neighbors:
                if nxt not in color:
                    continue
                if color[nxt] == WHITE:
                    color[nxt] = GRAY
                    path.append(nxt)
                    stack.append((nxt, iter(sorted(adj.get(nxt, [])))))
                    descended = True
                    break
                if color[nxt] == GRAY:
                    start = path.index(nxt)
                    return node, nxt, path[start:]
            if not descended:
                color[node] = BLACK
                path.pop()
                stack.pop()
    return None


def break_cycles(
    nodes: list[str], edges: list[dict]
) -> tuple[list[dict], list[dict]]:
    """Detect cycles via DFS and break them.

    For each back edge, the edge on the cycle whose SOURCE concept has the lower
    centrality (in+out degree) is removed; the back edge itself is the
    deterministic tie-breaker preference. Repeats until the graph is acyclic.
    Returns (remaining_edges, removed_edges).
    """
    edge_list = [dict(e) for e in edges]
    removed: list[dict] = []
    for _ in range(len(edge_list) + 1):
        adj: dict[str, list[str]] = {n: [] for n in nodes}
        for edge in edge_list:
            if edge["from"] in adj and edge["to"] in adj:
                if edge["to"] not in adj[edge["from"]]:
                    adj[edge["from"]].append(edge["to"])
        found = _find_back_edge(list(adj), adj)
        if found is None:
            break
        u, v, cycle_path = found
        cycle_edges = [
            (cycle_path[i], cycle_path[i + 1]) for i in range(len(cycle_path) - 1)
        ] + [(u, v)]
        score: dict[str, int] = {n: 0 for n in adj}
        for edge in edge_list:
            if edge["from"] in score:
                score[edge["from"]] += 1
            if edge["to"] in score:
                score[edge["to"]] += 1
        # Lowest source centrality first; prefer the back edge (u, v) on ties.
        victim = min(
            cycle_edges,
            key=lambda fe: (score.get(fe[0], 0), fe != (u, v), fe[0], fe[1]),
        )
        edge_list = [
            e
            for e in edge_list
            if not (e["from"] == victim[0] and e["to"] == victim[1])
        ]
        removed.append({"from": victim[0], "to": victim[1], "reason": "cycle"})
        logger.warning(
            "Stage 5: removed cycle edge %s -> %s (source centrality %d)",
            victim[0], victim[1], score.get(victim[0], 0),
        )
    return edge_list, removed


def transitive_prerequisites(
    nodes: list[str], edges: list[dict], max_depth: int = MAX_TRANSITIVE_DEPTH
) -> dict[str, list[str]]:
    """BFS ancestor closure up to max_depth for every node (deterministic order)."""
    adj: dict[str, list[str]] = {n: [] for n in nodes}
    for edge in edges:
        if edge["from"] in adj and edge["to"] in adj:
            if edge["to"] not in adj[edge["from"]]:
                adj[edge["from"]].append(edge["to"])
    for n in adj:
        adj[n].sort()

    closure: dict[str, list[str]] = {}
    for node in nodes:
        seen = {node}
        frontier = [node]
        collected: list[str] = []
        for _ in range(max_depth):
            level: list[str] = []
            for current in frontier:
                for prereq in adj.get(current, []):
                    if prereq not in seen:
                        seen.add(prereq)
                        level.append(prereq)
            level = sorted(set(level))
            collected.extend(level)
            if not level:
                break
            frontier = level
        closure[node] = collected
    return closure


async def run(taxonomy_path: Path, graph_path: Path, domain: str) -> None:
    """Run Stage 5 for one domain."""
    require_file(
        taxonomy_path,
        f"(run stage 3 first for domain '{domain}')",
    )
    taxonomy_doc = load_json_obj(taxonomy_path) or {}
    concepts: list[str] = []
    seen: set[str] = set()
    for item in taxonomy_doc.get("taxonomy", []):
        item = str(item).strip()
        if item and item not in seen:
            seen.add(item)
            concepts.append(item)

    if not concepts:
        logger.warning("Stage 5 [%s]: taxonomy is empty; writing empty graph", domain)
        save_json(
            graph_path,
            {
                "nodes": [],
                "edges": [],
                "leaves": [],
                "removed_cycles": [],
                "transitive_prerequisites": {},
            },
        )
        return

    progress_path = graph_path.with_name(graph_path.name + ".progress.json")
    raw_progress = load_json_obj(progress_path)
    progress: dict[str, dict] = {}
    if isinstance(raw_progress, dict):
        for key, value in raw_progress.items():
            if isinstance(value, dict):
                progress[str(key)] = value

    # Re-derive per-concept results from an existing graph so a clean re-run
    # makes zero LLM calls. Only concepts this stage actually processed as
    # targets ("processed") are trusted; prerequisite-only nodes are not.
    existing_graph = load_json_obj(graph_path)
    if isinstance(existing_graph, dict) and existing_graph.get("nodes"):
        edges_by_from: dict[str, list[str]] = {}
        for edge in existing_graph.get("edges", []):
            if isinstance(edge, dict) and edge.get("from") and edge.get("to"):
                edges_by_from.setdefault(str(edge["from"]), []).append(str(edge["to"]))
        processed = existing_graph.get("processed") or existing_graph.get("nodes")
        concept_set = set(concepts)
        seeded = 0
        for node in processed:
            if node not in concept_set or node in progress:
                continue
            prereq_names = edges_by_from.get(node, [])
            progress[node] = {
                "concept": node,
                "is_leaf": not prereq_names,
                "prerequisites": [{"name": p, "is_new": False} for p in prereq_names],
            }
            seeded += 1
        if seeded:
            logger.info(
                "Stage 5 [%s]: re-derived %d concept(s) from existing graph (no LLM calls)",
                domain, seeded,
            )

    pending = [c for c in concepts if c not in progress]

    # A graph written by a previously successful run (no progress file left
    # over) that already covers every concept is final: keep it untouched so
    # metadata (leaves, removed_cycles) is stable across re-runs.
    graph_is_current = (
        isinstance(existing_graph, dict)
        and bool(existing_graph.get("nodes"))
        and raw_progress is None
        and not pending
    )
    if graph_is_current:
        logger.info(
            "Stage 5 [%s]: existing graph already covers all %d concept(s); "
            "keeping it unchanged (no LLM calls)",
            domain, len(concepts),
        )
        return

    logger.info(
        "Stage 5 [%s]: %d concept(s), %d already processed, %d to process",
        domain, len(concepts), len(concepts) - len(pending), len(pending),
    )

    lock = asyncio.Lock()
    tax_lock = asyncio.Lock()
    counters = {"ok": 0, "failed": 0, "skipped": len(concepts) - len(pending)}
    pbar = tqdm(
        total=len(pending),
        desc=f"Stage 5 [{domain}] concept graph",
        unit="concept",
    )
    pbar.set_postfix(skip=counters["skipped"])

    async def register_new_concepts(names: list[str]) -> None:
        if not names:
            return
        added = 0
        async with tax_lock:
            for name in names:
                if name not in taxonomy_doc.get("merge_map", {}):
                    taxonomy_doc.setdefault("merge_map", {})[name] = name
                    taxonomy_doc.setdefault("taxonomy", []).append(name)
                    added += 1
            if added:
                save_json(taxonomy_path, taxonomy_doc)
        if added:
            logger.info(
                "Stage 5 [%s]: registered %d new prerequisite concept(s) in taxonomy",
                domain, added,
            )

    async def worker(concept: str) -> None:
        prompt_text = PROMPT_PREREQ.safe_substitute(
            domain=domain,
            concept=concept,
            all_concepts="\n".join(concepts),
        )
        try:
            data = await call_llm(prompt_text, expect_json=True)
            result = _coerce_result(concept, data)
        except LLMError as exc:
            logger.error(
                "Stage 5 [%s] concept %s: failed after all retries: %s "
                "(keeping it as a node with no prerequisite edges)",
                domain, concept, exc,
            )
            result = {"concept": concept, "is_leaf": False, "prerequisites": [], "failed": True}

        known = set(concepts)
        new_names = [
            p["name"] for p in result["prerequisites"] if p["name"] not in known
        ]
        if new_names:
            await register_new_concepts(new_names)

        async with lock:
            progress[concept] = result
            save_json(progress_path, progress)
            counters["ok" if not result.get("failed") else "failed"] += 1
            pbar.update(1)
            pbar.set_postfix(skip=counters["skipped"], **counters)

    if pending:
        await asyncio.gather(*(worker(c) for c in pending))
    pbar.close()

    # --- Assemble the DAG -----------------------------------------------------
    nodes = list(concepts)
    node_set = set(nodes)
    edges: list[dict] = []
    edge_set: set[tuple[str, str]] = set()
    leaves: list[str] = []
    for concept in concepts:
        result = progress.get(concept) or {}
        if result.get("failed"):
            continue  # not a genuine leaf — just an unprocessed node
        prereqs = result.get("prerequisites", [])
        if result.get("is_leaf") or not prereqs:
            leaves.append(concept)
        for prereq in prereqs[:MAX_PREREQUISITES]:
            name = prereq["name"]
            if name == concept:
                continue
            if name not in node_set:
                node_set.add(name)
                nodes.append(name)
            if (concept, name) not in edge_set:
                edge_set.add((concept, name))
                edges.append({"from": concept, "to": name})

    edges, removed_cycles = break_cycles(nodes, edges)
    closure = transitive_prerequisites(nodes, edges, max_depth=MAX_TRANSITIVE_DEPTH)

    graph = {
        "nodes": nodes,
        "edges": edges,
        "leaves": leaves,
        "removed_cycles": removed_cycles,
        "transitive_prerequisites": closure,
        # Bookkeeping for resumability: concepts processed as targets by this run.
        "processed": list(concepts),
    }
    save_json(graph_path, graph)
    if progress_path.exists():
        progress_path.unlink()
    logger.info(
        "Stage 5 [%s] complete: %d node(s), %d edge(s), %d leaf concept(s), "
        "%d cycle(s) removed",
        domain, len(nodes), len(edges), len(leaves), len(removed_cycles),
    )
