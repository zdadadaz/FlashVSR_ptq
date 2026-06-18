import json

from scripts.experiments.static_mixed_qat.render_leaderboard import render_html
from scripts.experiments.static_mixed_qat.update_leaderboard import build_leaderboard_row, write_jsonl_row


def test_build_leaderboard_row_hashes_policy_and_script(tmp_path):
    policy = tmp_path / "policy.json"
    policy.write_text('{"layers": {}}')
    reproduce = tmp_path / "reproduce.sh"
    reproduce.write_text("#!/usr/bin/env bash\necho hi\n")
    psnr = tmp_path / "psnr.json"
    psnr.write_text(json.dumps({"average_psnr": 28.5}))

    row = build_leaderboard_row(
        run_id="run1",
        policy=policy,
        reproduce_script=reproduce,
        psnr_json=psnr,
        a8_layers=3,
        a16_layers=1,
        activation_qdq_mode="static_tensor_symmetric",
        clipping="p99.9",
        qat=False,
    )

    assert row["policy_sha256"]
    assert row["reproduce_sha256"]
    assert row["psnr_vs_fp16_mean"] == 28.5
    assert row["a8_layers"] == 3
    assert row["a16_layers"] == 1


def test_write_jsonl_and_render_html(tmp_path):
    path = tmp_path / "leaderboard.jsonl"
    write_jsonl_row(path, {"run_id": "r1", "psnr_vs_fp16_mean": 20.0, "a16_layers": 2})
    write_jsonl_row(path, {"run_id": "r2", "psnr_vs_fp16_mean": 30.0, "a16_layers": 1})

    html = render_html(path)

    assert "r2" in html
    assert "r1" in html
    assert "FlashVSR Static Mixed QAT Leaderboard" in html
