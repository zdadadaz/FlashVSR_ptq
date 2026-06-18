import json

from scripts.experiments.static_mixed_qat.update_leaderboard import build_leaderboard_row


def test_leaderboard_reads_qat_summary_teacher_psnr(tmp_path):
    summary = tmp_path / "qat_summary.json"
    summary.write_text(json.dumps({"last_metrics": {"teacher_psnr_db": 13.25}}))

    row = build_leaderboard_row(run_id="qat", psnr_json=summary)

    assert row["psnr_vs_fp16_mean"] == 13.25
