"""Run Person A September QAT and validation clips.

Defaults are aligned with the user's requested environment/data:
- Python: FlashVSR_Integrated/.venv/bin/python
- QAT videos: datasets/train
- Validation inputs: data/lowres/{bowing,bus,carphone,city,coastguard}_*.mp4
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / ".venv" / "bin" / "python"
DEFAULT_CKPT = ROOT / "models" / "FlashVSR-v1.1" / "diffusion_pytorch_model_streaming_dmd.safetensors"
TEST_CLIPS = {
    "bowing": "bowing_cif.mp4",
    "bus": "bus_cif.mp4",
    "carphone": "carphone_qcif.mp4",
    "city": "city_cif.mp4",
    "coastguard": "coastguard_cif.mp4",
}


def run(cmd: list[str], log_path: Path, dry_run: bool = False) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(" ".join(cmd))
    if dry_run:
        log_path.write_text("DRY RUN\n" + " ".join(cmd) + "\n")
        return
    with log_path.open("w") as f:
        proc = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}); log={log_path}; cmd={' '.join(cmd)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="FlashVSR v1.1 September QAT from datasets/train videos + lowres eval")
    parser.add_argument("--run_id", default="2026-09-personA-flashvsr-v1.1-qat")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CKPT))
    parser.add_argument("--train_video_dir", default=str(ROOT / "datasets" / "train"))
    parser.add_argument("--output_root", default=str(ROOT / "outputs" / "qat" / "2026-09-personA"))
    parser.add_argument("--max_train_videos", type=int, default=16)
    parser.add_argument("--prepare_frames", type=int, default=4)
    parser.add_argument("--latent_size", default="16x16")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--smoke_steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--mode", default="a8w8", choices=["a8w8", "a16w8", "a8w4", "a16w4"])
    parser.add_argument("--activation_qdq_mode", default="dynamic_asymmetric", choices=["static_asymmetric", "dynamic_symmetric", "dynamic_asymmetric"])
    parser.add_argument("--eval_end_frame", type=int, default=16)
    parser.add_argument("--scale", type=int, default=4, choices=[2, 4])
    parser.add_argument("--smoke", action="store_true", help="Run one-step QAT and skip expensive video eval")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.output_root) / args.run_id
    sample_dir = run_dir / "samples"
    manifest = sample_dir / "manifest.jsonl"
    logs = run_dir / "logs"
    videos = run_dir / "videos"
    metrics = run_dir / "metrics"
    run_dir.mkdir(parents=True, exist_ok=True)

    prepare_cmd = [
        str(PYTHON), "scripts/qat/prepare_video_manifest.py",
        "--video_dir", args.train_video_dir,
        "--output_dir", str(sample_dir),
        "--manifest", str(manifest),
        "--max_videos", str(args.max_train_videos),
        "--frames", str(args.prepare_frames),
        "--latent_size", args.latent_size,
    ]
    run(prepare_cmd, logs / "prepare_manifest.log", dry_run=args.dry_run)

    qat_steps = args.smoke_steps if args.smoke else args.steps
    qat_out = run_dir / "train"
    qat_cmd = [
        str(PYTHON), "scripts/qat/finetune_fakequant_dit.py",
        "--checkpoint", args.checkpoint,
        "--manifest", str(manifest),
        "--output_dir", str(qat_out),
        "--mode", args.mode,
        "--activation_qdq_mode", args.activation_qdq_mode,
        "--steps", str(qat_steps),
        "--lr", str(args.lr),
        "--ema_decay", str(args.ema_decay),
        "--temporal_loss_weight", "0.05",
        "--target_psnr_drop_db", "0.4",
        "--gradient_checkpointing",
        "--dtype", "bf16",
        "--device", "cuda" if not args.smoke else "cuda",
    ]
    run(qat_cmd, logs / "qat_train.log", dry_run=args.dry_run)

    fakequant_ckpt = qat_out / "flashvsr_v1.1_qat_fakequant.pt"
    eval_rows = []
    if not args.smoke:
        for name, filename in TEST_CLIPS.items():
            inp = ROOT / "data" / "lowres" / filename
            fp16 = videos / f"{name}_fp16_first{args.eval_end_frame}.mp4"
            qat = videos / f"{name}_qat_first{args.eval_end_frame}.mp4"
            common = [
                str(PYTHON), "cli_main.py",
                "--input", str(inp),
                "--model", "FlashVSR-v1.1",
                "--vae_model", "Wan2.1",
                "--scale", str(args.scale),
                "--mode", "full",
                "--precision", "bf16",
                "--device", "cuda:0",
                "--attention_mode", "sdpa",
                "--start_frame", "0",
                "--end_frame", str(args.eval_end_frame),
                "--seed", "0",
            ]
            run(common + ["--output", str(fp16), "--quantize_mode", "None"], logs / f"eval_{name}_fp16.log", dry_run=args.dry_run)
            run(common + ["--output", str(qat), "--quantize_mode", "FakeQuant_A8W8", "--ckpt_path", str(fakequant_ckpt)], logs / f"eval_{name}_qat.log", dry_run=args.dry_run)
            psnr_json = metrics / f"{name}_fp16_vs_qat_psnr.json"
            run([str(PYTHON), "scripts/compare_video_psnr.py", str(fp16), str(qat), "--out-json", str(psnr_json)], logs / f"metric_{name}.log", dry_run=args.dry_run)
            eval_rows.append({"clip": name, "fp16": str(fp16), "qat": str(qat), "metric": str(psnr_json)})

    summary = {
        "run_id": args.run_id,
        "checkpoint": args.checkpoint,
        "train_video_dir": args.train_video_dir,
        "manifest": str(manifest),
        "qat_output_dir": str(qat_out),
        "fakequant_checkpoint": str(fakequant_ckpt),
        "test_clips": eval_rows,
        "smoke": args.smoke,
        "dry_run": args.dry_run,
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
