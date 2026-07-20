import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np
from skimage.metrics import structural_similarity


@dataclass
class CalibrationConfig:
    low_charge_ratio: float = 0.60
    baseline_percentile: float = 10.0
    regularization: float = 0.02
    max_iterations: int = 5000
    tolerance: float = 1e-6
    health_weight: float = 0.25
    minimum_health_score: float = 0.20
    maximum_scale: float = 5.0


def load_charge_map(path, key="inverted_charge_map"):
    path = Path(path)
    if path.suffix.lower() == ".npy":
        return np.load(path).squeeze().astype(float)
    if path.suffix.lower() == ".npz":
        with np.load(path) as data:
            return data[key].squeeze().astype(float)
    if path.suffix.lower() == ".mat":
        with h5py.File(path, "r") as data:
            return data[key][()].T.squeeze().astype(float)
    raise ValueError(f"Unsupported file type: {path.suffix}")


def detect_low_charge(healthy, degraded, config):
    baseline = np.abs(healthy)
    values = baseline[baseline > np.finfo(float).eps]
    if values.size == 0:
        return np.zeros_like(healthy, dtype=bool)
    floor = np.percentile(values, config.baseline_percentile)
    return (baseline >= floor) & (np.abs(degraded) < config.low_charge_ratio * baseline)


def laplacian_inpaint(image, mask, config):
    if not np.any(mask):
        return image.copy()
    output = image.copy().astype(float)
    known = ~mask
    output[mask] = np.median(image[known]) if np.any(known) else 0.0
    degree = np.zeros_like(output)
    degree[1:, :] += 1
    degree[:-1, :] += 1
    degree[:, 1:] += 1
    degree[:, :-1] += 1
    scale = max(np.max(np.abs(image)), 1.0)
    for _ in range(config.max_iterations):
        neighbours = np.zeros_like(output)
        neighbours[1:, :] += output[:-1, :]
        neighbours[:-1, :] += output[1:, :]
        neighbours[:, 1:] += output[:, :-1]
        neighbours[:, :-1] += output[:, 1:]
        candidate = (neighbours + config.regularization * image) / (
            degree + config.regularization
        )
        updated = output.copy()
        updated[mask] = candidate[mask]
        updated[known] = image[known]
        if np.max(np.abs(updated - output)) / scale < config.tolerance:
            return updated
        output = updated
    return output


def calculate_scale(healthy, restored, mask, health_score, config):
    valid = (~mask) & (np.abs(restored) > np.finfo(float).eps)
    if np.any(valid):
        reference_scale = np.median(
            np.abs(healthy[valid]) / np.maximum(np.abs(restored[valid]), np.finfo(float).eps)
        )
    else:
        reference_scale = 1.0
    health_scale = 1.0 / max(health_score, config.minimum_health_score)
    scale = np.exp(
        (1.0 - config.health_weight) * np.log(max(reference_scale, 1e-12))
        + config.health_weight * np.log(max(health_scale, 1e-12))
    )
    return float(np.clip(scale, 1.0 / config.maximum_scale, config.maximum_scale))


def nrmse(reference, estimate):
    error = np.sqrt(np.mean((estimate - reference) ** 2))
    scale = np.sqrt(np.mean(reference**2)) + np.finfo(float).eps
    return float(error / scale)


def evaluate(reference, estimate):
    data_range = max(reference.max(), estimate.max()) - min(reference.min(), estimate.min())
    score = structural_similarity(reference, estimate, data_range=max(data_range, 1e-12))
    return {"nrmse": nrmse(reference, estimate), "ssim": float(score)}


def calibrate(healthy, degraded, health_score, config=None):
    config = config or CalibrationConfig()
    healthy = np.asarray(healthy, dtype=float).squeeze()
    degraded = np.asarray(degraded, dtype=float).squeeze()
    if healthy.shape != degraded.shape or healthy.ndim != 2:
        raise ValueError("Healthy and degraded maps must be two-dimensional with equal shapes")
    if not 0.0 <= health_score <= 1.0:
        raise ValueError("health_score must be between 0 and 1")
    mask = detect_low_charge(healthy, degraded, config)
    restored = laplacian_inpaint(degraded, mask, config)
    scale = calculate_scale(healthy, restored, mask, health_score, config)
    calibrated = restored * scale
    metrics = {
        "scale_factor": scale,
        "abnormal_fraction": float(mask.mean()),
        "before": evaluate(healthy, degraded),
        "after": evaluate(healthy, calibrated),
    }
    return calibrated, mask, metrics


def fit_pressure_calibration(charge, pressure):
    charge = np.asarray(charge, dtype=float).ravel()
    pressure = np.asarray(pressure, dtype=float).ravel()
    valid = np.isfinite(charge) & np.isfinite(pressure)
    matrix = np.column_stack((charge[valid], np.ones(valid.sum())))
    gain, intercept = np.linalg.lstsq(matrix, pressure[valid], rcond=None)[0]
    return float(gain), float(intercept)


def charge_to_pressure(charge_map, gain, intercept):
    return np.maximum(gain * np.asarray(charge_map, dtype=float) + intercept, 0.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--healthy", required=True)
    parser.add_argument("--degraded", required=True)
    parser.add_argument("--health-score", required=True, type=float)
    parser.add_argument("--output", default="calibration_result.npz")
    parser.add_argument("--metrics", default="calibration_metrics.json")
    parser.add_argument("--key", default="inverted_charge_map")
    args = parser.parse_args()
    config = CalibrationConfig()
    healthy = load_charge_map(args.healthy, args.key)
    degraded = load_charge_map(args.degraded, args.key)
    calibrated, mask, metrics = calibrate(healthy, degraded, args.health_score, config)
    np.savez_compressed(
        args.output,
        calibrated_charge_map=calibrated,
        abnormal_mask=mask,
    )
    payload = {"config": asdict(config), "metrics": metrics}
    Path(args.metrics).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
