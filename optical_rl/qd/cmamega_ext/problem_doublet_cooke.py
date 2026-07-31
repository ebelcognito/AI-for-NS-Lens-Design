from __future__ import annotations

import copy
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np

from .descriptors import DESCRIPTOR_NAMES, build_descriptor_dict
from .types import EvalResult


Bounds6D = Tuple[
    Tuple[float, float],
    Tuple[float, float],
    Tuple[float, float],
    Tuple[float, float],
    Tuple[float, float],
    Tuple[float, float],
]


def project_box(theta: np.ndarray, bounds: Sequence[Tuple[float, float]]) -> np.ndarray:
    arr = np.asarray(theta, dtype=np.float64).reshape(-1)
    lo = np.array([b[0] for b in bounds], dtype=np.float64)
    hi = np.array([b[1] for b in bounds], dtype=np.float64)
    return np.clip(arr, lo, hi)


def theta_to_unit(theta: np.ndarray, bounds: Sequence[Tuple[float, float]]) -> np.ndarray:
    theta = project_box(theta, bounds)
    lo = np.array([b[0] for b in bounds], dtype=np.float64)
    hi = np.array([b[1] for b in bounds], dtype=np.float64)
    center = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo)
    return (theta - center) / np.where(half > 0.0, half, 1.0)


def unit_to_theta(unit: np.ndarray, bounds: Sequence[Tuple[float, float]]) -> np.ndarray:
    u = np.clip(np.asarray(unit, dtype=np.float64).reshape(-1), -1.0, 1.0)
    lo = np.array([b[0] for b in bounds], dtype=np.float64)
    hi = np.array([b[1] for b in bounds], dtype=np.float64)
    center = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo)
    return project_box(center + u * half, bounds)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _deeplens_root() -> Path:
    return _repo_root() / "DeepLens"


def _ensure_deeplens_path() -> None:
    root = str(_deeplens_root())
    if root not in sys.path:
        sys.path.insert(0, root)


def _latin_hypercube(n: int, dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    if n <= 0:
        return np.zeros((0, dim), dtype=np.float64)
    out = np.zeros((n, dim), dtype=np.float64)
    for j in range(dim):
        perm = rng.permutation(n)
        out[:, j] = (perm + rng.random(n)) / float(n)
    return out


def _disk_samples(num_rays: int) -> np.ndarray:
    idx = np.arange(num_rays, dtype=np.float64) + 0.5
    radius = np.sqrt(idx / float(max(num_rays, 1)))
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    theta = idx * golden_angle
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    return np.stack([x, y], axis=-1)


@dataclass(frozen=True)
class DoubletCookeConfig:
    lens_path: str = "/home/hisham26/Desktop/RL/optical_rl/qd/doublet_starter.json"
    object_depth_mm: float = -1000.0
    wavelength_um: float = 0.587
    theta_bounds: Bounds6D = (
        (-0.05, 0.05),
        (-0.05, 0.05),
        (-0.05, 0.05),
        (-0.05, 0.05),
        (2.5, 7.5),
        (5.0, 120.0),
    )
    invalid_tol: float = 0.05
    invalid_penalty_weight: float = 5.0
    num_rays: int = 256
    alpha_deg: float = 0.5
    stage2_field_deg: float = 2.0
    device: str = "auto"
    active_descriptor_indices: Tuple[int, int] = (0, 1)


class FixedCookeDoubletProblem:
    """Cooke-derived fixed-topology doublet with retained stop."""

    def __init__(self, config: DoubletCookeConfig):
        self.config = config
        self._backend_loaded = False
        self._torch = None
        self._GeoLens = None
        self._Ray = None
        self._device = None
        self._template_lens = None
        self._aperture_xy = None
        self._refractive_idx = (0, 1, 3, 4)
        self._stop_idx = 2
        self._thickness_1 = None
        self._thickness_2 = None
        self._stop_d = None
        self._gap_base = None
        self._delta_n = None

    def descriptor_names(self) -> Sequence[str]:
        return DESCRIPTOR_NAMES

    def _ensure_backend(self) -> None:
        if self._backend_loaded:
            return

        _ensure_deeplens_path()
        import torch
        from deeplens.optics import GeoLens
        from deeplens.optics.light import Ray

        if self.config.device == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        else:
            device = torch.device(self.config.device)

        self._torch = torch
        self._GeoLens = GeoLens
        self._Ray = Ray
        self._device = device
        self._template_lens = self._load_template_lens()
        aperture = _disk_samples(int(self.config.num_rays)) * float(
            self._template_lens.surfaces[self._refractive_idx[0]].r
        )
        self._aperture_xy = torch.tensor(aperture, dtype=torch.float32, device=self._device)
        self._backend_loaded = True

    def _load_template_lens(self):
        torch = self._torch
        lens = self._GeoLens(filename=self.config.lens_path, device=self._device)

        # Preferred path: starter JSON already has the retained 5-surface block.
        # Backward compatibility: allow full Cooke by slicing indices [2..6].
        if len(lens.surfaces) >= 7:
            keep = [2, 3, 4, 5, 6]
        else:
            keep = list(range(len(lens.surfaces)))
        kept = [copy.deepcopy(lens.surfaces[i]) for i in keep]
        if len(kept) != 5:
            raise ValueError(
                f"Expected 5 surfaces in starter block, got {len(kept)} from {self.config.lens_path}"
            )
        d0 = float(kept[0].d.item())
        for s in kept:
            s.d = torch.tensor(float(s.d.item()) - d0, dtype=s.d.dtype, device=self._device)
        lens.surfaces = kept
        lens.d_sensor = torch.tensor(float(lens.d_sensor.item()) - d0, dtype=lens.d_sensor.dtype, device=self._device)

        for idx in self._refractive_idx:
            lens.surfaces[idx].c = torch.tensor(0.0, dtype=lens.surfaces[idx].c.dtype, device=self._device)
        lens.post_computation()

        s0 = lens.surfaces[self._refractive_idx[0]]
        s1 = lens.surfaces[self._refractive_idx[1]]
        s2 = lens.surfaces[self._refractive_idx[2]]
        s3 = lens.surfaces[self._refractive_idx[3]]
        stop = lens.surfaces[self._stop_idx]
        self._thickness_1 = float(s1.d.item() - s0.d.item())
        self._thickness_2 = float(s3.d.item() - s2.d.item())
        self._stop_d = float(stop.d.item())
        self._gap_base = float(s2.d.item() - s1.d.item())

        delta_n = []
        n_prev = 1.0
        for idx in self._refractive_idx:
            n_next = float(lens.surfaces[idx].mat2.n)
            delta_n.append(n_next - n_prev)
            n_prev = n_next
        self._delta_n = torch.tensor(delta_n, dtype=torch.float32, device=self._device)
        return lens

    def initial_theta(self) -> np.ndarray:
        self._ensure_backend()
        return np.array([0.0, 0.0, 0.0, 0.0, float(self._gap_base), float(self._template_lens.d_sensor.item())])

    def sample_theta(self, n: int, seed: int) -> np.ndarray:
        unit = _latin_hypercube(int(n), 6, int(seed))
        lo = np.array([b[0] for b in self.config.theta_bounds], dtype=np.float64)
        hi = np.array([b[1] for b in self.config.theta_bounds], dtype=np.float64)
        return lo + unit * (hi - lo)

    def project_theta(self, theta: np.ndarray) -> np.ndarray:
        return project_box(theta, self.config.theta_bounds)

    def _build_lens_from_theta_tensor(self, theta_t):
        lens = copy.deepcopy(self._template_lens)
        s0, s1, s2, s3 = self._refractive_idx
        stop = self._stop_idx
        lens.surfaces[s0].c = theta_t[0]
        lens.surfaces[s1].c = theta_t[1]
        lens.surfaces[s2].c = theta_t[2]
        lens.surfaces[s3].c = theta_t[3]

        d0 = float(lens.surfaces[s0].d.item())
        lens.surfaces[s1].d = lens.surfaces[s0].d + float(self._thickness_1)
        lens.surfaces[stop].d = d0 + float(self._stop_d)
        lens.surfaces[s2].d = lens.surfaces[s1].d + theta_t[4]
        lens.surfaces[s3].d = lens.surfaces[s2].d + float(self._thickness_2)
        lens.d_sensor = theta_t[5]
        return lens

    def _make_object_point(self, field_x_deg: float, field_y_deg: float):
        depth = float(self.config.object_depth_mm)
        x = depth * math.tan(math.radians(field_x_deg))
        y = depth * math.tan(math.radians(field_y_deg))
        return x, y, depth

    def _trace_field(self, lens, field_x_deg: float, field_y_deg: float):
        torch = self._torch
        x_obj, y_obj, z_obj = self._make_object_point(field_x_deg, field_y_deg)
        ray_count = self._aperture_xy.shape[0]
        dtype = lens.surfaces[self._refractive_idx[0]].c.dtype
        origins = torch.empty((ray_count, 3), dtype=dtype, device=self._device)
        origins[:, 0] = x_obj
        origins[:, 1] = y_obj
        origins[:, 2] = z_obj

        targets = torch.empty((ray_count, 3), dtype=dtype, device=self._device)
        targets[:, :2] = self._aperture_xy.to(dtype=dtype)
        targets[:, 2] = float(lens.surfaces[self._refractive_idx[0]].d.item())
        ray = self._Ray(origins, targets - origins, wvln=float(self.config.wavelength_um), device=self._device)
        return lens.trace2sensor(ray)

    def _field_points(self) -> Tuple[Tuple[float, float], ...]:
        f = float(self.config.stage2_field_deg)
        return ((0.0, 0.0), (f, 0.0), (-f, 0.0), (0.0, f), (0.0, -f))

    def _rms_spot(self, ray):
        torch = self._torch
        valid = ray.is_valid
        xy = ray.o[..., :2]
        denom = valid.sum().clamp_min(1e-8)
        centroid = (xy * valid.unsqueeze(-1)).sum(dim=0) / denom
        err2 = ((xy - centroid) ** 2).sum(dim=-1)
        rms = ((err2 * valid).sum() / denom + 1e-12).sqrt()
        return rms, 1.0 - (valid.sum() / max(float(valid.numel()), 1.0))

    def _offaxis_aberr_proxy(self, ray):
        torch = self._torch
        valid = ray.is_valid
        xy = ray.o[..., :2]
        denom = valid.sum().clamp_min(1e-8)
        mean = (xy * valid.unsqueeze(-1)).sum(dim=0) / denom
        d = xy - mean
        x = d[:, 0]
        y = d[:, 1]
        xx = (x * x * valid).sum() / denom
        yy = (y * y * valid).sum() / denom
        xy_cov = (x * y * valid).sum() / denom
        trace = xx + yy
        det = xx * yy - xy_cov * xy_cov
        disc = torch.clamp(trace * trace - 4.0 * det, min=1e-12).sqrt()
        lmax = 0.5 * (trace + disc)
        lmin = 0.5 * (trace - disc)
        return (lmax - lmin) / (lmax + lmin + 1e-8)

    def _surface_powers(self, theta_t):
        return self._delta_n * theta_t[:4]

    def _evaluate_core(self, theta_t):
        torch = self._torch
        lens = self._build_lens_from_theta_tensor(theta_t)

        rms_terms = []
        invalid_fracs = []
        for fx, fy in self._field_points():
            ray = self._trace_field(lens, fx, fy)
            rms, invalid_fraction = self._rms_spot(ray)
            rms_terms.append(rms)
            invalid_fracs.append(invalid_fraction)

        rms_stack = torch.stack(rms_terms)
        rms_mean = rms_stack.mean()
        invalid_fraction_mean = torch.stack(invalid_fracs).mean()
        invalid_penalty = torch.clamp(invalid_fraction_mean - float(self.config.invalid_tol), min=0.0)
        fitness = -rms_mean - float(self.config.invalid_penalty_weight) * invalid_penalty

        alpha = float(self.config.alpha_deg)
        offaxis = self._trace_field(lens, 0.0, alpha)
        valid = offaxis.is_valid
        denom = valid.sum().clamp_min(1e-8)
        mean_y = (offaxis.o[..., 1] * valid).sum() / denom
        aberr_proxy = self._offaxis_aberr_proxy(offaxis)
        phi = self._surface_powers(theta_t)
        desc_dict = build_descriptor_dict(
            mean_offaxis_y=mean_y,
            alpha_deg=alpha,
            phi=phi,
            aberr_proxy=aberr_proxy,
        )
        desc_tensors = [desc_dict[name] for name in DESCRIPTOR_NAMES]
        desc_vector = torch.stack(desc_tensors)
        active = tuple(int(i) for i in self.config.active_descriptor_indices)
        measures = (desc_vector[active[0]], desc_vector[active[1]])

        metrics = {
            "rms_spot": float(rms_mean.detach().cpu().item()),
            "invalid_fraction": float(invalid_fraction_mean.detach().cpu().item()),
            "invalid_penalty": float(invalid_penalty.detach().cpu().item()),
            "field_rms_center": float(rms_stack[0].detach().cpu().item()),
            "field_rms_edge_x_pos": float(rms_stack[1].detach().cpu().item()),
            "field_rms_edge_x_neg": float(rms_stack[2].detach().cpu().item()),
            "field_rms_edge_y_pos": float(rms_stack[3].detach().cpu().item()),
            "field_rms_edge_y_neg": float(rms_stack[4].detach().cpu().item()),
            "d0_power_proxy": float(desc_vector[0].detach().cpu().item()),
            "d1_power_split_front_rear": float(desc_vector[1].detach().cpu().item()),
            "d2_total_power_mag": float(desc_vector[2].detach().cpu().item()),
            "d3_offaxis_aberr_proxy": float(desc_vector[3].detach().cpu().item()),
            "foclen": float(getattr(lens, "foclen", float("nan"))),
        }
        return fitness, measures, desc_vector, metrics

    def evaluate(self, theta: np.ndarray, with_grads: bool = True) -> EvalResult:
        self._ensure_backend()
        torch = self._torch
        theta_raw = np.asarray(theta, dtype=np.float64).reshape(6)
        theta_proj = self.project_theta(theta_raw)
        clipped = np.not_equal(theta_raw, theta_proj)
        theta_t = torch.tensor(theta_proj, dtype=torch.float32, device=self._device, requires_grad=with_grads)
        fitness, measures_t, desc_t, metrics = self._evaluate_core(theta_t)

        metrics["clip_hits_total"] = float(clipped.sum())
        for i, hit in enumerate(clipped.tolist()):
            metrics[f"clip_hit_dim_{i}"] = float(1.0 if hit else 0.0)

        grads = None
        if with_grads:
            g_f = torch.autograd.grad(fitness, theta_t, retain_graph=True, allow_unused=False)[0]
            grads = {"fitness": g_f.detach().cpu().numpy().astype(np.float64, copy=False)}
            for idx in range(desc_t.shape[0]):
                g_d = torch.autograd.grad(desc_t[idx], theta_t, retain_graph=True, allow_unused=False)[0]
                grads[f"d{idx}"] = g_d.detach().cpu().numpy().astype(np.float64, copy=False)

        return EvalResult(
            theta=theta_proj.astype(np.float64, copy=False),
            fitness=float(fitness.detach().cpu().item()),
            measures=(
                float(measures_t[0].detach().cpu().item()),
                float(measures_t[1].detach().cpu().item()),
            ),
            descriptor_vector=desc_t.detach().cpu().numpy().astype(np.float64, copy=False),
            metrics=metrics,
            grads=grads,
        )
