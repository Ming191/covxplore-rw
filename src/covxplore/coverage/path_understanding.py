"""Metrics that measure whether path tracing improves behavioral understanding.

Coverage % alone is a weak proxy. These metrics ask: did the model correctly
predict control flow, hit its stated target, and repair after divergence feedback?
"""

from __future__ import annotations

from typing import Iterable, Sequence

from covxplore.coverage.path_comparison import (
    compare_expected_path,
    effective_expected_path,
    polarity_to_bool,
)
from covxplore.types.test_result import TestResult


def _path_bearing(results: Iterable[TestResult]) -> list[TestResult]:
    return [result for result in results if effective_expected_path(result)]


def target_hit(result: TestResult) -> bool | None:
    """Whether this run observed the final expected_path / target polarity."""
    path = effective_expected_path(result)
    if not path:
        return None
    target = path[-1]
    want_true = polarity_to_bool(target.polarity)
    for branch in result.unvisited_branches:
        if branch.node_id != target.node_id:
            continue
        observed = branch.true_visited if want_true else branch.false_visited
        # Target hit if polarity observed OR node absent from unvisited (both sides seen).
        if observed:
            return True
        if not branch.true_visited and not branch.false_visited:
            return False
        return False
    # Node not listed as unvisited ⇒ both sides observed this run ⇒ target side hit.
    return True


def compute_path_understanding_metrics(
    *,
    accepted: Sequence[TestResult],
    rejected: Sequence[TestResult] | None = None,
    known_node_ids: set[int] | None = None,
) -> dict:
    """Aggregate path-understanding stats across all executed candidates with a path."""
    pool = list(accepted) + list(rejected or [])
    bearing = _path_bearing(pool)
    if not bearing:
        return {
            "path_candidates": 0,
            "path_match_rate": None,
            "path_divergence_rate": None,
            "target_hit_rate": None,
            "catalog_id_rate": None,
            "repair_success_rate": None,
            "matched": 0,
            "diverged": 0,
            "target_hits": 0,
            "target_misses": 0,
            "repairs_attempted": 0,
            "repairs_succeeded": 0,
        }

    matched = diverged = 0
    target_hits = target_misses = 0
    catalog_ok = catalog_total = 0
    divergence_events: list[tuple[int, str]] = []  # (node_id, polarity)

    for result in bearing:
        comparison = compare_expected_path(result, known_node_ids=known_node_ids)
        if comparison.matched is True:
            matched += 1
        elif comparison.matched is False:
            diverged += 1
            step = comparison.divergence_step
            if step is not None:
                divergence_events.append((step.node_id, step.polarity))

        hit = target_hit(result)
        if hit is True:
            target_hits += 1
        elif hit is False:
            target_misses += 1

        path = effective_expected_path(result)
        if known_node_ids is not None and path:
            catalog_total += len(path)
            catalog_ok += sum(1 for step in path if step.node_id in known_node_ids)

    # Repair: after a divergence on (node, polarity), a later accepted test hits that target.
    repairs_attempted = 0
    repairs_succeeded = 0
    accepted_bearing = _path_bearing(accepted)
    for index, result in enumerate(bearing):
        comparison = compare_expected_path(result, known_node_ids=known_node_ids)
        if comparison.matched is not False or comparison.divergence_step is None:
            continue
        node_id = comparison.divergence_step.node_id
        polarity = comparison.divergence_step.polarity
        repairs_attempted += 1
        later = bearing[index + 1 :]
        # Prefer later accepted tests when available.
        later_accepted = [
            item
            for item in later
            if item in accepted_bearing or item.test_name in {t.test_name for t in accepted_bearing}
        ]
        search = later_accepted or later
        succeeded = False
        for follow in search:
            path = effective_expected_path(follow)
            if not path:
                continue
            last = path[-1]
            if last.node_id == node_id and last.polarity == polarity and target_hit(follow) is True:
                succeeded = True
                break
            # Also count covering the divergent polarity even if not last step.
            for step in path:
                if step.node_id == node_id and step.polarity == polarity and target_hit(follow) is True:
                    succeeded = True
                    break
            if succeeded:
                break
        if succeeded:
            repairs_succeeded += 1

    total = matched + diverged
    target_total = target_hits + target_misses
    return {
        "path_candidates": len(bearing),
        "matched": matched,
        "diverged": diverged,
        "path_match_rate": round(matched / total, 4) if total else None,
        "path_divergence_rate": round(diverged / total, 4) if total else None,
        "target_hits": target_hits,
        "target_misses": target_misses,
        "target_hit_rate": round(target_hits / target_total, 4) if target_total else None,
        "catalog_id_rate": round(catalog_ok / catalog_total, 4) if catalog_total else None,
        "repairs_attempted": repairs_attempted,
        "repairs_succeeded": repairs_succeeded,
        "repair_success_rate": (
            round(repairs_succeeded / repairs_attempted, 4) if repairs_attempted else None
        ),
    }
