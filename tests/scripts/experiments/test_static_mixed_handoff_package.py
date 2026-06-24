import json
from pathlib import Path

from scripts.experiments.static_mixed_qat.build_npu_handoff_package import (
    build_handoff_manifest,
    sync_leaderboard_to_daily,
)


def test_handoff_manifest_marks_reds30_training_run_and_gt_ready(tmp_path):
    lb = tmp_path / "leaderboard.jsonl"
    ckpt = tmp_path / "qat.pt"
    policy = tmp_path / "policy.json"
    manifest = tmp_path / "manifest.jsonl"
    for p in (ckpt, policy, manifest):
        p.write_text(p.name)
    lb.write_text(json.dumps({
        "schema_version": "flashvsr.static_mixed_leaderboard_row.v1",
        "run_id": "reds30_qat",
        "checkpoint": str(ckpt),
        "policy": str(policy),
        "manifest": str(manifest),
        "qat": True,
        "observer": "observer_freeze",
        "psnr_vs_fp16_mean": 22.5,
        "eval_set": "REDS30-training-smoke",
        "notes": "training_dataset=REDS30",
    }) + "\n")
    gt = tmp_path / "gt"
    (gt / "LQ").mkdir(parents=True)
    (gt / "GT").mkdir()
    (gt / "LQ" / "000.mp4").write_text("lq")
    (gt / "GT" / "000.mp4").write_text("gt")

    out = build_handoff_manifest(leaderboard=lb, leaderboard_html=None, gt_dir=gt)

    assert out["best_vs_fp16_run_id"] == "reds30_qat"
    assert out["gt_readiness"]["status"] == "ready"
    assert out["artifacts"][0]["training_dataset"] == "REDS30"


def test_sync_leaderboard_to_daily_copies_jsonl_and_html(tmp_path):
    lb = tmp_path / "leaderboard.jsonl"
    html = tmp_path / "leaderboard.html"
    daily = tmp_path / "daily"
    lb.write_text('{"run_id":"x"}\n')
    html.write_text("<html></html>")

    copied = sync_leaderboard_to_daily(lb, html, daily, prefix="20260618")

    assert Path(copied["jsonl"]).read_text() == lb.read_text()
    assert Path(copied["html"]).read_text() == html.read_text()
    assert Path(copied["jsonl"]).name == "20260618_flashvsr_static_mixed_leaderboard.jsonl"
