import csv
import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"


def run_command(args):
    print("\n$ " + " ".join(str(item) for item in args), flush=True)
    completed = subprocess.run(
        [str(item) for item in args],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(completed.stdout, flush=True)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}: {' '.join(str(item) for item in args)}")
    return completed.stdout


def metric_value(pattern, text):
    match = re.search(pattern, text)
    if not match:
        raise ValueError(f"Could not parse metric with pattern: {pattern}")
    return float(match.group(1))


def measure(dataset, gt_mean, args):
    output = run_command([sys.executable, "measure.py", *args])
    return {
        "dataset": dataset,
        "gt_mean": gt_mean,
        "psnr": metric_value(r"Avg\.PSNR:\s*([0-9.]+)", output),
        "ssim": metric_value(r"Avg\.SSIM:\s*([0-9.]+)", output),
        "lpips": metric_value(r"Avg\.LPIPS:\s*([0-9.]+)", output),
    }


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    eval_runs = [
        [
            sys.executable,
            "eval.py",
            "--lol",
            "--weights_path",
            "weights/lolv1_litesafe_best_composite.pth",
            "--lite_mode",
            "safe",
            "--semantic_in_channels",
            "384",
            "--semantic_prior_dir",
            "priors/lolv1_eval/semantic",
            "--depth_prior_dir",
            "priors/lolv1_eval/depth",
            "--normal_prior_dir",
            "priors/lolv1_eval/normal",
            "--semantic_quality_dir",
            "priors/lolv1_eval/semantic_quality",
            "--depth_quality_dir",
            "priors/lolv1_eval/depth_quality",
            "--normal_quality_dir",
            "priors/lolv1_eval/normal_quality",
            "--strict_prior_loading",
            "--tta_mode",
            "flip4",
            "--num_workers",
            "0",
        ],
        [
            sys.executable,
            "eval.py",
            "--lol_v2_syn",
            "--weights_path",
            "weights/lolv2_syn_litesafe_best_composite.pth",
            "--lite_mode",
            "safe",
            "--semantic_in_channels",
            "384",
            "--semantic_prior_dir",
            "priors/lolv2_syn_eval/semantic",
            "--depth_prior_dir",
            "priors/lolv2_syn_eval/depth",
            "--normal_prior_dir",
            "priors/lolv2_syn_eval/normal",
            "--semantic_quality_dir",
            "priors/lolv2_syn_eval/semantic_quality",
            "--depth_quality_dir",
            "priors/lolv2_syn_eval/depth_quality",
            "--normal_quality_dir",
            "priors/lolv2_syn_eval/normal_quality",
            "--strict_prior_loading",
            "--tta_mode",
            "flip4",
            "--num_workers",
            "0",
        ],
        [
            sys.executable,
            "eval.py",
            "--lol_v2_real",
            "--weights_path",
            "weights/lolv2_real_best_composite.pth",
            "--alpha",
            "0.8",
            "--num_workers",
            "0",
        ],
    ]

    for command in eval_runs:
        run_command(command)

    rows = [
        measure("LOLv1", False, ["--lol"]),
        measure("LOLv1", True, ["--lol", "--use_GT_mean"]),
        measure("LOLv2-real", False, ["--lol_v2_real"]),
        measure("LOLv2-real", True, ["--lol_v2_real", "--use_GT_mean"]),
        measure("LOLv2-syn", False, ["--lol_v2_syn"]),
        measure("LOLv2-syn", True, ["--lol_v2_syn", "--use_GT_mean"]),
    ]

    json_path = RESULTS_DIR / "reproduce_metrics.json"
    csv_path = RESULTS_DIR / "reproduce_metrics.csv"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", "gt_mean", "psnr", "ssim", "lpips"])
        writer.writeheader()
        writer.writerows(rows)

    print("\n| Dataset | GT mean | PSNR | SSIM | LPIPS |")
    print("| --- | --- | ---: | ---: | ---: |")
    for row in rows:
        gt_mean = "Yes" if row["gt_mean"] else "No"
        print(f"| {row['dataset']} | {gt_mean} | {row['psnr']:.4f} | {row['ssim']:.4f} | {row['lpips']:.4f} |")

    print(f"\nSaved: {json_path}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
