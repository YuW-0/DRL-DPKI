import argparse
import csv
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

from parameter import GlobalConfig
from main import ExperimentManager

def get_output_root() -> Path:
    configured = os.environ.get("DRL_DPKI_OUTPUT_DIR")
    if configured:
        return Path(configured)
    legacy = Path("/mnt/data/wy2024")
    if legacy.is_dir():
        return legacy
    repo_root = Path(__file__).resolve().parents[1]
    return repo_root / "outputs"


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _backup_json(result_json: Path, backup_dir: Path, scale_label: str, malicious_label: str, startup_label: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"ca_scale_sensitivity_results_scale_{scale_label}_{malicious_label}_{startup_label}_{timestamp}.json"
    backup_path = backup_dir / backup_name
    shutil.copy2(result_json, backup_path)
    return backup_path


def _load_existing_csv(csv_path: Path) -> dict[int, float]:
    existing = {}
    if not csv_path.exists():
        return existing

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                scale = int(row.get("ca_scale", "").strip())
                rate = float(row.get("final_malicious_ca_selection_rate_percent", "").strip())
            except Exception:
                continue
            existing[scale] = rate
    return existing


def _extract_final_malicious_selection_rate_percent(
    data: dict,
    scale: int,
    target_ratio: int = 50,
) -> Optional[float]:
    test_info = data.get("test_info", {})
    malicious_ratios = test_info.get("malicious_ratios")
    if not isinstance(malicious_ratios, list) or target_ratio not in malicious_ratios:
        raise ValueError(f"test_info.malicious_ratios is missing {target_ratio}")

    results = data.get("results", {})
    scale_data = results.get(str(scale))
    if not isinstance(scale_data, dict):
        return None

    malicious_counts = scale_data.get("malicious_counts", [])
    if not isinstance(malicious_counts, list):
        return None

    target_count = int(scale * target_ratio / 100)
    if target_count not in malicious_counts:
        return None

    metrics_list = scale_data.get("metrics", [])
    if not isinstance(metrics_list, list):
        return None

    target_index = malicious_counts.index(target_count)
    if target_index >= len(metrics_list):
        return None

    metrics = metrics_list[target_index]
    if not isinstance(metrics, dict):
        return None

    rate = metrics.get("malicious_na_selection_rate")
    if not isinstance(rate, (int, float)):
        return None

    return round(rate * 100.0, 2)


def main() -> int:
    config = GlobalConfig()
    default_result_json = (
        get_output_root()
        / "baseline_methods"
        / "strategy_comparison"
        / "ca_scale_sensitivity"
        / "ca_scale_sensitivity_results.json"
    )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-json",
        type=str,
        default=str(default_result_json),
    )
    parser.add_argument("--start-scale", type=int, default=100)
    parser.add_argument("--end-scale", type=int, default=3000)
    parser.add_argument("--step", type=int, default=100)
    parser.add_argument("--time-points", type=int, default=config.training.N_TIME_POINTS)
    parser.add_argument("--seed", type=int, default=config.training.RANDOM_SEED)
    parser.add_argument("--resume-csv", type=str, default=None)
    args = parser.parse_args()

    result_json = Path(args.result_json).resolve()
    results_dir = result_json.parent
    results_dir.mkdir(parents=True, exist_ok=True)

    malicious_selection_mode = "random"
    warmup_first_round = True
    malicious_label = "random"
    startup_label = "warm"

    if result_json.exists():
        existing_scale_label = "preexisting"
        try:
            existing_data = _load_json(result_json)
            scales = existing_data.get("test_info", {}).get("scales", [])
            if isinstance(scales, list) and len(scales) == 1:
                existing_scale_label = str(scales[0])
            elif isinstance(scales, list) and len(scales) > 1:
                existing_scale_label = "multi"
        except Exception:
            existing_scale_label = "preexisting"
        _backup_json(result_json, results_dir, existing_scale_label, malicious_label, startup_label)

    resume_csv = Path(args.resume_csv).resolve() if args.resume_csv else (
        results_dir / "final_malicious_selection_rate_at_50pct_random_warm_100-2000_step100.csv"
    )
    existing_results = _load_existing_csv(resume_csv)
    output_by_scale: dict[int, float] = dict(existing_results)

    if existing_results:
        max_existing_scale = max(existing_results.keys())
        run_start_scale = max(args.start_scale, max_existing_scale + args.step)
    else:
        run_start_scale = args.start_scale

    for scale in range(run_start_scale, args.end_scale + 1, args.step):
        manager = ExperimentManager(
            config_name="CA_SCALE_SENSITIVITY",
            save_dir=str(results_dir),
        )
        manager.run_ca_scale_sensitivity(
            n_time_points=args.time_points,
            scales=[scale],
            seed=args.seed,
            malicious_selection_mode=malicious_selection_mode,
            warmup_first_round=warmup_first_round,
        )

        if not result_json.exists():
            raise FileNotFoundError(f"Result file was not generated: {result_json}")

        data = _load_json(result_json)
        rate_percent = _extract_final_malicious_selection_rate_percent(data, scale, target_ratio=50)
        _backup_json(result_json, results_dir, str(scale), malicious_label, startup_label)

        if rate_percent is None:
            continue

        output_by_scale[scale] = rate_percent

    csv_path = results_dir / (
        f"final_malicious_selection_rate_at_50pct_random_warm_{args.start_scale}-{args.end_scale}_step{args.step}.csv"
    )
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["ca_scale", "final_malicious_ca_selection_rate_percent"],
        )
        writer.writeheader()
        for scale in sorted(output_by_scale.keys()):
            writer.writerow(
                {
                    "ca_scale": scale,
                    "final_malicious_ca_selection_rate_percent": output_by_scale[scale],
                }
            )

    print(f"CSV generated: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
