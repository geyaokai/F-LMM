from __future__ import annotations

from pathlib import Path

from scripts.demo.correction_selfcheck import (
    apply_human_overrides,
    build_all_snapshots,
    build_override_template,
    validate_snapshot,
)


def sample_report():
    return {
        "summary": {
            "manifest_path": "scripts/demo/manifests/stability_cases.v1.json",
        },
        "samples": [
            {
                "sample_id": "0000_demo_case",
                "source_id": "demo_case",
                "status": "ok",
                "question": "What is the color of the man's shirt?",
                "notes": "demo",
                "raw_answer": "The man's shirt appears to be yellow.",
                "answer": "The man's shirt appears to be yellow.",
                "roi_answer": None,
                "auto_failure_tags": [],
                "manual_failure_tags": [],
                "phrases": [
                    {
                        "index": 0,
                        "text": "man's shirt",
                        "mention_text": "man's shirt",
                        "mention_char_span": [4, 15],
                        "mention_token_span": [1, 3],
                        "concept_text": "shirt",
                        "concept_char_span": [10, 15],
                        "concept_token_span": [2, 3],
                        "concept_lemma": "shirt",
                        "concept_source": "heuristic_head",
                    },
                    {
                        "index": 1,
                        "text": "yellow",
                        "mention_text": "yellow",
                        "mention_char_span": [32, 38],
                        "mention_token_span": [5, 6],
                        "concept_text": "yellow",
                        "concept_char_span": [32, 38],
                        "concept_token_span": [5, 6],
                        "concept_lemma": "yellow",
                        "concept_source": "heuristic_head",
                    },
                ],
                "ground_records": [
                    {
                        "index": 0,
                        "phrase": "man's shirt",
                        "char_span": [10, 15],
                        "token_span": [2, 3],
                        "bbox": [160, 32, 234, 114],
                        "overlay_path": "overlay.png",
                        "mask_path": "mask.png",
                        "roi_path": "roi.png",
                    }
                ],
            }
        ],
    }


def test_build_snapshots_and_auto_repair():
    baseline, auto_snapshot, human_snapshot, override_errors = build_all_snapshots(
        sample_report(),
        report_path=Path("/tmp/report.json"),
        overrides=None,
    )
    assert human_snapshot is None
    assert override_errors == []

    baseline_concepts = baseline["samples"][0]["concept_records"]
    auto_concepts = auto_snapshot["samples"][0]["concept_records"]

    assert baseline_concepts[0]["judge_label"] == "keep"
    assert baseline_concepts[1]["judge_label"] == "reject"
    assert auto_concepts[0]["judge_label"] == "repair"
    assert auto_concepts[0]["corrected_concept_text"] == "shirt"
    assert auto_concepts[1]["judge_label"] == "reject"


def test_human_override_merge_and_validation():
    _, auto_snapshot, _, _ = build_all_snapshots(
        sample_report(),
        report_path=Path("/tmp/report.json"),
        overrides=None,
    )
    override_template = build_override_template(auto_snapshot)
    first_row = dict(override_template[0])
    first_row.update(
        {
            "judge_label": "keep",
            "judge_reason": "human confirmed",
            "corrected_concept_text": "",
            "corrected_answer_span": [],
            "corrected_bbox": [],
            "comment": "looks correct",
        }
    )
    merged, errors = apply_human_overrides(auto_snapshot, [first_row])
    assert errors == []
    concept = merged["samples"][0]["concept_records"][0]
    assert concept["judge_label"] == "keep"
    assert concept["judge_source"] == "human"

    summary = validate_snapshot(merged)
    assert summary["failure_counts"]["human_override_failure"] == 0
    assert summary["judge_counts"]["keep"] >= 1
