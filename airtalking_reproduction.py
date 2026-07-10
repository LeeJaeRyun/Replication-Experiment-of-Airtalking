from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


POLICIES = ("Stochastic", "LinUCB", "SA", "Greedy", "MCTS")
OPT_POLICIES = ("LinUCB", "SA", "Greedy", "MCTS")
AREAS = (100, 200, 300, 400, 500)


@dataclass(frozen=True)
class SemanticCompressionMode:
    name: str
    rho_c: float
    quality: float


@dataclass(frozen=True)
class SemanticProfile:
    name: str
    strategy: str
    modes: Tuple[SemanticCompressionMode, ...] = ()
    target_thresholds: Tuple[Tuple[float, float], ...] = (
        (-15.0, 0.80),
        (-10.0, 0.88),
        (-5.0, 0.925),
        (0.0, 0.95),
        (float("inf"), 0.965),
    )


@dataclass(frozen=True)
class PaperParams:
    # Table III values.
    n_uav: int = 20
    n_device: int = 20
    dt: float = 1.0
    t_slots: int = 1000
    repeats: int = 10
    vmax_uav: float = 23.0
    accel_uav: float = 3.0
    decel_uav: float = 2.0
    height_mean: float = 20.0
    carrier_bandwidth: float = 80e6
    carrier_frequency: float = 5e9
    p_uav_tx: float = 0.2
    p_device_tx: float = 0.1
    noise_psd: float = 4e-21
    noise_figure_db: float = 5.0
    alpha_u2u: float = 2.2
    alpha_u2g: float = 2.7
    k_u2u_db: float = 10.0
    rho_c: float = 0.104
    rho_r: float = 3.0
    enc_bitrate: float = 91.30e6
    dec_bitrate: float = 23.23e6


@dataclass(frozen=True)
class AssumedParams:
    # These values are not numerically specified in the paper.
    request_probability: float = 0.020
    device_diffusion: float = 0.9
    device_speed_cap: float = 1.4
    workload_mean_bits: float = 420e6
    workload_std_bits: float = 90e6
    workload_min_bits: float = 180e6
    workload_max_bits: float = 760e6
    p_move: float = 680.0
    p_hover: float = 610.0
    p_encode: float = 0.9
    p_decode: float = 0.9
    p_d2d_radio: float = 0.2
    energy_weight: float = 1.0 / 12000.0
    density_interference_scale: float = 25.0
    linucb_alpha: float = 0.55
    linucb_lambda: float = 1.0
    linucb_candidate_samples: int = 0
    sa_iterations: int = 36
    sa_temperature: float = 3.0
    sa_cooling: float = 0.90
    mcts_samples: int = 80
    random_semantic_encode_probability: float = 2.0 / 3.0
    random_semantic_decode_probability: float = 0.5
    seed: int = 260707


def apply_assumed_overrides(assumed: AssumedParams, overrides: Sequence[str]) -> AssumedParams:
    values = dict(assumed.__dict__)
    defaults = assumed.__dict__
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Expected --assumed KEY=VALUE, got {item!r}")
        key, raw_value = item.split("=", 1)
        key = key.strip()
        if key not in values:
            valid = ", ".join(sorted(values))
            raise ValueError(f"Unknown assumed parameter {key!r}. Valid keys: {valid}")
        default_value = defaults[key]
        if isinstance(default_value, bool):
            value = raw_value.lower() in {"1", "true", "yes", "on"}
        elif isinstance(default_value, int) and not isinstance(default_value, bool):
            value = int(raw_value)
        else:
            value = float(raw_value)
        values[key] = value
    return AssumedParams(**values)


def apply_semantic_summary(
    paper: PaperParams,
    summary_path: Optional[Path],
    raw_basis: str,
    encoder_mode: str,
    decoder_mode: str,
    profile_kind: str,
) -> Tuple[PaperParams, Dict[str, object]]:
    if summary_path is None:
        return paper, {"source": "paper_table_iii", "applied": False}
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if profile_kind == "zlib":
        rho_key = "rho_c_uncompressed_mean" if raw_basis == "uncompressed" else "rho_c_png_mean"
        enc_key = "encode_bitrate_mbps_median"
        dec_key = "decode_bitrate_mbps_median"
    elif profile_kind == "feature":
        rho_key = "rho_c_feature_uncompressed_mean" if raw_basis == "uncompressed" else "rho_c_feature_png_mean"
        enc_key = "feature_encode_bitrate_mbps_median"
        dec_key = "feature_decode_bitrate_mbps_median"
    else:
        raise ValueError(f"unknown profile kind {profile_kind}")
    rho_c = float(summary[rho_key])
    rho_r = float(summary.get("rho_r_proxy", paper.rho_r))
    if encoder_mode == "measured":
        enc_bitrate = float(summary["encode_bitrate_mbps_median"]) * 1e6
    elif encoder_mode == "paper":
        enc_bitrate = paper.enc_bitrate
    else:
        raise ValueError(f"unknown encoder mode {encoder_mode}")
    if decoder_mode == "measured":
        dec_bitrate = float(summary[dec_key]) * 1e6
    elif decoder_mode == "paper":
        dec_bitrate = paper.dec_bitrate
    else:
        raise ValueError(f"unknown decoder mode {decoder_mode}")
    updated = PaperParams(
        **{
            **paper.__dict__,
            "rho_c": rho_c,
            "rho_r": rho_r,
            "enc_bitrate": enc_bitrate,
            "dec_bitrate": dec_bitrate,
        }
    )
    return updated, {
        "source": str(summary_path),
        "applied": True,
        "raw_basis": raw_basis,
        "profile_kind": profile_kind,
        "encoder_mode": encoder_mode,
        "decoder_mode": decoder_mode,
        "rho_c": rho_c,
        "rho_r": rho_r,
        "enc_bitrate": enc_bitrate,
        "dec_bitrate": dec_bitrate,
        "num_samples": summary.get("num_samples"),
        "palette_classes": summary.get("palette_classes"),
    }


@dataclass
class Candidate:
    tx: int
    rx: int
    sem_tx: int
    sem_rx: int
    duration: float
    t_move_tx: float
    t_move_rx: float
    t_hover: float
    t_d2d: float
    t_encode: float
    t_decode: float
    e_flight: float
    e_nonflight: float
    travel_distance: float
    sinr_db: float
    rate_mbps: float
    cost: float
    tx_target: np.ndarray
    rx_target: np.ndarray
    workload_bits: float
    semantic_mode: str = "raw"
    semantic_ratio: float = 1.0
    semantic_quality: float = 1.0


@dataclass
class ActiveAction:
    remaining: float
    tx: int
    rx: int
    source: int
    dest: int
    tx_target: np.ndarray
    rx_target: np.ndarray
    candidate: Candidate


@dataclass
class SimulationResult:
    finished: np.ndarray
    flight_energy_per_req: np.ndarray
    nonflight_energy_per_req: np.ndarray
    avg_time: np.ndarray
    avg_travel: np.ndarray
    encodes: np.ndarray
    decodes: np.ndarray
    semantic_quality: np.ndarray
    semantic_payload_ratio: np.ndarray
    sinr_samples: List[float]
    summary: Dict[str, float]


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = (
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def travel_time(distance: float, paper: PaperParams) -> float:
    if distance <= 1e-9:
        return 0.0
    a = paper.accel_uav
    d = paper.decel_uav
    vmax = paper.vmax_uav
    d_to_vmax = vmax * vmax / (2.0 * a) + vmax * vmax / (2.0 * d)
    if distance >= d_to_vmax:
        return vmax / a + vmax / d + (distance - d_to_vmax) / vmax
    v_peak = math.sqrt(2.0 * distance / (1.0 / a + 1.0 / d))
    return v_peak / a + v_peak / d


def reflected_step(pos: np.ndarray, area: float, rng: np.random.Generator, assumed: AssumedParams, dt: float) -> np.ndarray:
    sigma = math.sqrt(2.0 * assumed.device_diffusion * dt)
    step = rng.normal(0.0, sigma, size=2)
    norm = float(np.linalg.norm(step))
    max_step = assumed.device_speed_cap * dt
    if norm > max_step:
        step *= max_step / norm
    out = pos.copy()
    out[:2] += step
    for axis in (0, 1):
        if out[axis] < 0:
            out[axis] = -out[axis]
        if out[axis] > area:
            out[axis] = 2 * area - out[axis]
        out[axis] = min(area, max(0.0, out[axis]))
    return out


def channel_gain(distance: float, alpha: float, k_linear: float, rng: Optional[np.random.Generator] = None) -> float:
    distance = max(distance, 1.0)
    c = 299_792_458.0
    fc = PaperParams().carrier_frequency
    wavelength = c / fc
    g0 = (wavelength / (4.0 * math.pi)) ** 2
    if rng is None:
        fading_power = 1.0
    else:
        sigma = math.sqrt(1.0 / (2.0 * (k_linear + 1.0)))
        los = math.sqrt(k_linear / (k_linear + 1.0))
        real = los + rng.normal(0.0, sigma)
        imag = rng.normal(0.0, sigma)
        fading_power = real * real + imag * imag
    return g0 * fading_power / (distance ** alpha)


def link_sinr(
    tx_pos: np.ndarray,
    rx_pos: np.ndarray,
    active: Sequence[ActiveAction],
    device_positions: np.ndarray,
    paper: PaperParams,
    density_interference_scale: float,
    rng: Optional[np.random.Generator] = None,
) -> float:
    k = 10 ** (paper.k_u2u_db / 10.0)
    signal_distance = float(np.linalg.norm(tx_pos - rx_pos))
    signal = paper.p_uav_tx * channel_gain(signal_distance, paper.alpha_u2u, k, rng)
    interference = 0.0
    for action in active:
        interference += paper.p_uav_tx * channel_gain(float(np.linalg.norm(action.tx_target - rx_pos)), paper.alpha_u2u, k, rng)
        source_pos = device_positions[action.source]
        interference += paper.p_device_tx * channel_gain(float(np.linalg.norm(source_pos - rx_pos)), paper.alpha_u2g, k, rng)
    noise_factor = 10 ** (paper.noise_figure_db / 10.0)
    noise = noise_factor * paper.noise_psd * paper.carrier_bandwidth
    span_x = max(1.0, float(np.max(device_positions[:, 0]) - np.min(device_positions[:, 0])))
    span_y = max(1.0, float(np.max(device_positions[:, 1]) - np.min(device_positions[:, 1])))
    span = max(10.0, max(span_x, span_y))
    # Equation (18) includes surrounding UAV and ground interference. Only active
    # actions are explicit in the event simulator, so this density term preserves
    # the paper's strong small-area interference behavior.
    density_penalty = 1.0 + density_interference_scale * (100.0 / span) ** 2
    return max(signal / ((interference + noise) * density_penalty), 1e-12)


def target_above(device_pos: np.ndarray, uav_index: int, paper: PaperParams) -> np.ndarray:
    centered = uav_index - (paper.n_uav - 1) / 2.0
    height = paper.height_mean + 0.5 * centered
    return np.array([device_pos[0], device_pos[1], height], dtype=float)


def workload_sample(rng: np.random.Generator, assumed: AssumedParams) -> float:
    value = rng.normal(assumed.workload_mean_bits, assumed.workload_std_bits)
    return float(np.clip(value, assumed.workload_min_bits, assumed.workload_max_bits))


def semantic_choice(
    sem_tx: int,
    sinr_db: float,
    paper: PaperParams,
    profile: Optional[SemanticProfile],
) -> Tuple[str, float, float]:
    if not sem_tx:
        return "raw", 1.0, 1.0
    if profile is None:
        return "fixed", paper.rho_c, 1.0
    if profile.strategy == "fixed":
        if profile.modes:
            mode = profile.modes[0]
            return mode.name, mode.rho_c, mode.quality
        return profile.name, paper.rho_c, 1.0
    if profile.strategy != "adaptive":
        raise ValueError(f"Unknown semantic profile strategy {profile.strategy!r}")
    if not profile.modes:
        return profile.name, paper.rho_c, 1.0

    target_quality = profile.target_thresholds[-1][1]
    for upper_sinr, required_quality in profile.target_thresholds:
        if sinr_db < upper_sinr:
            target_quality = required_quality
            break
    for mode in sorted(profile.modes, key=lambda item: item.rho_c):
        if mode.quality >= target_quality:
            return mode.name, mode.rho_c, mode.quality
    mode = max(profile.modes, key=lambda item: item.quality)
    return mode.name, mode.rho_c, mode.quality


def make_candidate(
    tx: int,
    rx: int,
    sem_tx: int,
    sem_rx: int,
    source: int,
    dest: int,
    workload_bits: float,
    uav_positions: np.ndarray,
    device_positions: np.ndarray,
    active: Sequence[ActiveAction],
    paper: PaperParams,
    assumed: AssumedParams,
    rng: Optional[np.random.Generator] = None,
    semantic_profile: Optional[SemanticProfile] = None,
) -> Candidate:
    tx_target = target_above(device_positions[source], tx, paper)
    rx_target = target_above(device_positions[dest], rx, paper)
    d_tx = float(np.linalg.norm(uav_positions[tx] - tx_target))
    d_rx = float(np.linalg.norm(uav_positions[rx] - rx_target))
    t_move_tx = travel_time(d_tx, paper)
    t_move_rx = travel_time(d_rx, paper)
    sinr = link_sinr(tx_target, rx_target, active, device_positions, paper, assumed.density_interference_scale, rng)
    rate = paper.carrier_bandwidth * math.log2(1.0 + sinr)
    sinr_db = 10.0 * math.log10(max(sinr, 1e-12))
    semantic_mode, semantic_ratio, semantic_quality = semantic_choice(sem_tx, sinr_db, paper, semantic_profile)
    payload_bits = workload_bits * (semantic_ratio if sem_tx else 1.0)
    phi = max(0.15, min(1.0, workload_bits / assumed.workload_max_bits))
    t_encode = sem_tx * workload_bits * phi / paper.enc_bitrate
    t_decode = sem_tx * sem_rx * semantic_ratio * paper.rho_r * workload_bits / paper.dec_bitrate
    t_d2d = payload_bits / max(rate, 1e-9)
    t_hover = t_d2d + t_encode + t_decode
    duration = max(t_move_tx, t_move_rx) + t_hover
    e_flight = assumed.p_move * (t_move_tx + t_move_rx) + 2.0 * assumed.p_hover * t_hover
    e_nonflight = assumed.p_d2d_radio * t_d2d + assumed.p_encode * t_encode + assumed.p_decode * t_decode
    cost = duration + assumed.energy_weight * (e_flight + e_nonflight)
    travel_distance = d_tx + d_rx
    return Candidate(
        tx=tx,
        rx=rx,
        sem_tx=sem_tx,
        sem_rx=sem_rx,
        duration=duration,
        t_move_tx=t_move_tx,
        t_move_rx=t_move_rx,
        t_hover=t_hover,
        t_d2d=t_d2d,
        t_encode=t_encode,
        t_decode=t_decode,
        e_flight=e_flight,
        e_nonflight=e_nonflight,
        travel_distance=travel_distance,
        sinr_db=sinr_db,
        rate_mbps=rate / 1e6,
        cost=cost,
        tx_target=tx_target,
        rx_target=rx_target,
        workload_bits=workload_bits,
        semantic_mode=semantic_mode,
        semantic_ratio=semantic_ratio,
        semantic_quality=semantic_quality,
    )


def enumerate_candidates(
    available: Sequence[int],
    source: int,
    dest: int,
    workload_bits: float,
    uav_positions: np.ndarray,
    device_positions: np.ndarray,
    active: Sequence[ActiveAction],
    paper: PaperParams,
    assumed: AssumedParams,
    semantic_enabled: bool,
    semantic_profile: Optional[SemanticProfile] = None,
) -> List[Candidate]:
    states = [(0, 0)] if not semantic_enabled else [(0, 0), (1, 0), (1, 1)]
    out: List[Candidate] = []
    for tx in available:
        for rx in available:
            if tx == rx:
                continue
            for sem_tx, sem_rx in states:
                out.append(
                    make_candidate(
                        tx,
                        rx,
                        sem_tx,
                        sem_rx,
                        source,
                        dest,
                        workload_bits,
                        uav_positions,
                        device_positions,
                        active,
                        paper,
                        assumed,
                        semantic_profile=semantic_profile,
                    )
                )
    return out


class LinUCB:
    def __init__(self, assumed: AssumedParams):
        self.dim = 7
        self.alpha = assumed.linucb_alpha
        self.a = assumed.linucb_lambda * np.eye(self.dim)
        self.b = np.zeros(self.dim)

    @staticmethod
    def features(c: Candidate, area: float, assumed: AssumedParams) -> np.ndarray:
        travel_norm = min(3.0, c.travel_distance / max(area * math.sqrt(2.0), 1.0))
        time_norm = min(3.0, c.duration / 40.0)
        flight_norm = min(3.0, c.e_flight / 30000.0)
        nonflight_norm = min(3.0, c.e_nonflight / 12.0)
        return np.array(
            [
                1.0,
                1.0 / (1.0 + travel_norm),
                1.0 / (1.0 + time_norm),
                1.0 / (1.0 + flight_norm),
                1.0 / (1.0 + nonflight_norm),
                1.0 if c.sem_tx else 0.0,
                c.workload_bits / assumed.workload_max_bits,
            ],
            dtype=float,
        )

    def select(self, candidates: Sequence[Candidate], area: float, assumed: AssumedParams) -> Candidate:
        inv_a = np.linalg.inv(self.a)
        theta = inv_a @ self.b
        best_score = -1e18
        best = candidates[0]
        for c in candidates:
            x = self.features(c, area, assumed)
            score = float(theta @ x + self.alpha * math.sqrt(max(0.0, x @ inv_a @ x)))
            if score > best_score:
                best_score = score
                best = c
        return best

    def update(self, candidate: Candidate, area: float, assumed: AssumedParams) -> None:
        x = self.features(candidate, area, assumed)
        normalized_cost = candidate.cost / 40.0
        reward = 1.0 / (1.0 + normalized_cost)
        self.a += np.outer(x, x)
        self.b += reward * x


def policy_select(
    policy: str,
    candidates: Sequence[Candidate],
    rng: np.random.Generator,
    assumed: AssumedParams,
    linucb: Optional[LinUCB],
    area: float,
) -> Candidate:
    if policy == "Stochastic":
        return candidates[int(rng.integers(0, len(candidates)))]
    if policy == "Greedy":
        return min(candidates, key=lambda c: c.cost)
    if policy == "SA":
        current = candidates[int(rng.integers(0, len(candidates)))]
        best = current
        temp = assumed.sa_temperature
        for _ in range(assumed.sa_iterations):
            proposal = candidates[int(rng.integers(0, len(candidates)))]
            delta = proposal.cost - current.cost
            if delta <= 0 or rng.random() < math.exp(-delta / max(temp, 1e-9)):
                current = proposal
                if current.cost < best.cost:
                    best = current
            temp *= assumed.sa_cooling
        return best
    if policy == "MCTS":
        sample_count = min(len(candidates), assumed.mcts_samples)
        indexes = rng.choice(len(candidates), size=sample_count, replace=False)
        sampled = [candidates[int(i)] for i in indexes]
        # UCT-like optimism: prefer low cost, with a mild reward for semantic exploration.
        return min(sampled, key=lambda c: c.cost - (0.35 if c.sem_tx else 0.0))
    if policy == "LinUCB":
        if linucb is None:
            raise ValueError("LinUCB state missing")
        return linucb.select(candidates, area, assumed)
    raise ValueError(f"Unknown policy {policy}")


def random_state(rng: np.random.Generator, semantic_enabled: bool, assumed: AssumedParams) -> Tuple[int, int]:
    if not semantic_enabled:
        return 0, 0
    if rng.random() >= assumed.random_semantic_encode_probability:
        return 0, 0
    decode = 1 if rng.random() < assumed.random_semantic_decode_probability else 0
    return 1, decode


def random_candidate(
    available_uavs: Sequence[int],
    source: int,
    dest: int,
    workload: float,
    uav_positions: np.ndarray,
    device_positions: np.ndarray,
    active: Sequence[ActiveAction],
    paper: PaperParams,
    assumed: AssumedParams,
    semantic_enabled: bool,
    rng: np.random.Generator,
    semantic_profile: Optional[SemanticProfile] = None,
) -> Candidate:
    tx = available_uavs[int(rng.integers(0, len(available_uavs)))]
    rx_choices = [u for u in available_uavs if u != tx]
    rx = rx_choices[int(rng.integers(0, len(rx_choices)))]
    sem_tx, sem_rx = random_state(rng, semantic_enabled, assumed)
    return make_candidate(
        tx,
        rx,
        sem_tx,
        sem_rx,
        source,
        dest,
        workload,
        uav_positions,
        device_positions,
        active,
        paper,
        assumed,
        semantic_profile=semantic_profile,
    )


def sampled_candidates(
    count: int,
    available_uavs: Sequence[int],
    source: int,
    dest: int,
    workload: float,
    uav_positions: np.ndarray,
    device_positions: np.ndarray,
    active: Sequence[ActiveAction],
    paper: PaperParams,
    assumed: AssumedParams,
    semantic_enabled: bool,
    rng: np.random.Generator,
    semantic_profile: Optional[SemanticProfile] = None,
) -> List[Candidate]:
    out: List[Candidate] = []
    seen = set()
    max_combinations = len(available_uavs) * max(0, len(available_uavs) - 1) * (3 if semantic_enabled else 1)
    attempts = min(max_combinations * 2, max(count * 4, count + 10))
    for _ in range(attempts):
        tx = available_uavs[int(rng.integers(0, len(available_uavs)))]
        rx_choices = [u for u in available_uavs if u != tx]
        rx = rx_choices[int(rng.integers(0, len(rx_choices)))]
        sem_tx, sem_rx = random_state(rng, semantic_enabled, assumed)
        key = (tx, rx, sem_tx, sem_rx)
        if key in seen:
            continue
        seen.add(key)
        out.append(
            make_candidate(
                tx,
                rx,
                sem_tx,
                sem_rx,
                source,
                dest,
                workload,
                uav_positions,
                device_positions,
                active,
                paper,
                assumed,
                semantic_profile=semantic_profile,
            )
        )
        if len(out) >= count:
            break
    if not out:
        out.append(
            random_candidate(
                available_uavs,
                source,
                dest,
                workload,
                uav_positions,
                device_positions,
                active,
                paper,
                assumed,
                semantic_enabled,
                rng,
                semantic_profile,
            )
        )
    return out


def select_candidate_for_policy(
    policy: str,
    available_uavs: Sequence[int],
    source: int,
    dest: int,
    workload: float,
    uav_positions: np.ndarray,
    device_positions: np.ndarray,
    active: Sequence[ActiveAction],
    paper: PaperParams,
    assumed: AssumedParams,
    semantic_enabled: bool,
    rng: np.random.Generator,
    linucb: Optional[LinUCB],
    area: float,
    semantic_profile: Optional[SemanticProfile] = None,
) -> Candidate:
    if policy == "Stochastic":
        return random_candidate(
            available_uavs,
            source,
            dest,
            workload,
            uav_positions,
            device_positions,
            active,
            paper,
            assumed,
            semantic_enabled,
            rng,
            semantic_profile,
        )
    if policy == "SA":
        current = random_candidate(
            available_uavs,
            source,
            dest,
            workload,
            uav_positions,
            device_positions,
            active,
            paper,
            assumed,
            semantic_enabled,
            rng,
            semantic_profile,
        )
        best = current
        temp = assumed.sa_temperature
        for _ in range(assumed.sa_iterations):
            proposal = random_candidate(
                available_uavs,
                source,
                dest,
                workload,
                uav_positions,
                device_positions,
                active,
                paper,
                assumed,
                semantic_enabled,
                rng,
                semantic_profile,
            )
            delta = proposal.cost - current.cost
            if delta <= 0 or rng.random() < math.exp(-delta / max(temp, 1e-9)):
                current = proposal
                if current.cost < best.cost:
                    best = current
            temp *= assumed.sa_cooling
        return best
    if policy == "MCTS":
        candidates = sampled_candidates(
            assumed.mcts_samples,
            available_uavs,
            source,
            dest,
            workload,
            uav_positions,
            device_positions,
            active,
            paper,
            assumed,
            semantic_enabled,
            rng,
            semantic_profile,
        )
        return min(candidates, key=lambda c: c.cost - (0.35 if c.sem_tx else 0.0))
    if policy == "LinUCB" and assumed.linucb_candidate_samples > 0:
        candidates = sampled_candidates(
            assumed.linucb_candidate_samples,
            available_uavs,
            source,
            dest,
            workload,
            uav_positions,
            device_positions,
            active,
            paper,
            assumed,
            semantic_enabled,
            rng,
            semantic_profile,
        )
    else:
        candidates = enumerate_candidates(
            available_uavs,
            source,
            dest,
            workload,
            uav_positions,
            device_positions,
            active,
            paper,
            assumed,
            semantic_enabled,
            semantic_profile,
        )
    return policy_select(policy, candidates, rng, assumed, linucb, area)


def run_single(
    area: int,
    policy: str,
    repeat: int,
    semantic_enabled: bool,
    paper: PaperParams,
    assumed: AssumedParams,
    semantic_profile: Optional[SemanticProfile] = None,
) -> SimulationResult:
    seed = assumed.seed + area * 97 + repeat * 1009 + (0 if semantic_enabled else 500_000) + POLICIES.index(policy) * 100_003
    rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)

    heights = np.array([paper.height_mean + 0.5 * (i - (paper.n_uav - 1) / 2.0) for i in range(paper.n_uav)])
    device_positions = np.column_stack(
        [rng.uniform(0, area, paper.n_device), rng.uniform(0, area, paper.n_device), np.zeros(paper.n_device)]
    ).astype(float)
    uav_positions = np.column_stack(
        [rng.uniform(0, area, paper.n_uav), rng.uniform(0, area, paper.n_uav), heights]
    ).astype(float)

    uav_busy = np.zeros(paper.n_uav, dtype=bool)
    device_busy = np.zeros(paper.n_device, dtype=bool)
    active: List[ActiveAction] = []
    linucb = LinUCB(assumed) if policy == "LinUCB" else None

    finished_series = np.zeros(paper.t_slots + 1)
    flight_series = np.zeros(paper.t_slots + 1)
    nonflight_series = np.zeros(paper.t_slots + 1)
    avg_time_series = np.zeros(paper.t_slots + 1)
    avg_travel_series = np.zeros(paper.t_slots + 1)
    encode_series = np.zeros(paper.t_slots + 1)
    decode_series = np.zeros(paper.t_slots + 1)
    semantic_quality_series = np.zeros(paper.t_slots + 1)
    semantic_payload_ratio_series = np.zeros(paper.t_slots + 1)

    completed = 0
    cumulative_flight = 0.0
    cumulative_nonflight = 0.0
    cumulative_time = 0.0
    cumulative_travel = 0.0
    cumulative_encodes = 0
    cumulative_decodes = 0
    cumulative_semantic_quality = 0.0
    cumulative_semantic_payload_ratio = 0.0
    semantic_mode_counts: Dict[str, float] = {"raw": 0.0, "fixed": 0.0}
    if semantic_profile is not None:
        semantic_mode_counts[semantic_profile.name] = 0.0
        for mode in semantic_profile.modes:
            semantic_mode_counts[mode.name] = 0.0
    sinr_samples: List[float] = []

    for t in range(1, paper.t_slots + 1):
        for action in list(active):
            action.remaining -= paper.dt
            if action.remaining <= 1e-9:
                active.remove(action)
                uav_busy[action.tx] = False
                uav_busy[action.rx] = False
                device_busy[action.source] = False
                device_busy[action.dest] = False
                uav_positions[action.tx] = action.tx_target
                uav_positions[action.rx] = action.rx_target

                c = action.candidate
                completed += 1
                cumulative_flight += c.e_flight
                cumulative_nonflight += c.e_nonflight
                cumulative_time += c.duration
                cumulative_travel += c.travel_distance
                cumulative_encodes += int(c.sem_tx)
                cumulative_decodes += int(c.sem_rx)
                cumulative_semantic_quality += c.semantic_quality
                cumulative_semantic_payload_ratio += c.semantic_ratio
                semantic_mode_counts[c.semantic_mode] = semantic_mode_counts.get(c.semantic_mode, 0.0) + 1.0
                sinr_samples.append(c.sinr_db)
                if linucb is not None:
                    linucb.update(c, area, assumed)

        for d in range(paper.n_device):
            if not device_busy[d]:
                device_positions[d] = reflected_step(device_positions[d], float(area), rng, assumed, paper.dt)

        order = list(range(paper.n_device))
        py_rng.shuffle(order)
        for source in order:
            if device_busy[source]:
                continue
            if rng.random() > assumed.request_probability:
                continue
            free_devices = [i for i in range(paper.n_device) if i != source and not device_busy[i]]
            available_uavs = [i for i in range(paper.n_uav) if not uav_busy[i]]
            if len(free_devices) == 0 or len(available_uavs) < 2:
                continue
            dest = free_devices[int(rng.integers(0, len(free_devices)))]
            workload = workload_sample(rng, assumed)
            selected = select_candidate_for_policy(
                policy,
                available_uavs,
                source,
                dest,
                workload,
                uav_positions,
                device_positions,
                active,
                paper,
                assumed,
                semantic_enabled,
                rng,
                linucb,
                float(area),
                semantic_profile,
            )
            noisy_selected = make_candidate(
                selected.tx,
                selected.rx,
                selected.sem_tx,
                selected.sem_rx,
                source,
                dest,
                workload,
                uav_positions,
                device_positions,
                active,
                paper,
                assumed,
                rng,
                semantic_profile,
            )
            uav_busy[noisy_selected.tx] = True
            uav_busy[noisy_selected.rx] = True
            device_busy[source] = True
            device_busy[dest] = True
            active.append(
                ActiveAction(
                    remaining=noisy_selected.duration,
                    tx=noisy_selected.tx,
                    rx=noisy_selected.rx,
                    source=source,
                    dest=dest,
                    tx_target=noisy_selected.tx_target,
                    rx_target=noisy_selected.rx_target,
                    candidate=noisy_selected,
                )
            )

        denom = max(completed, 1)
        finished_series[t] = completed
        flight_series[t] = cumulative_flight / denom
        nonflight_series[t] = cumulative_nonflight / denom
        avg_time_series[t] = cumulative_time / denom
        avg_travel_series[t] = cumulative_travel / denom
        encode_series[t] = cumulative_encodes
        decode_series[t] = cumulative_decodes
        semantic_quality_series[t] = cumulative_semantic_quality / denom
        semantic_payload_ratio_series[t] = cumulative_semantic_payload_ratio / denom

    summary = {
        "finished": float(completed),
        "flight_energy_per_req": cumulative_flight / max(completed, 1),
        "nonflight_energy_per_req": cumulative_nonflight / max(completed, 1),
        "avg_time": cumulative_time / max(completed, 1),
        "avg_travel": cumulative_travel / max(completed, 1),
        "encodes": float(cumulative_encodes),
        "decodes": float(cumulative_decodes),
        "semantic_quality": cumulative_semantic_quality / max(completed, 1),
        "semantic_payload_ratio": cumulative_semantic_payload_ratio / max(completed, 1),
        "sinr_median_db": float(statistics.median(sinr_samples)) if sinr_samples else float("nan"),
    }
    for mode_name, count in semantic_mode_counts.items():
        safe_name = mode_name.replace("-", "_").replace(" ", "_")
        summary[f"mode_{safe_name}_count"] = float(count)
    return SimulationResult(
        finished=finished_series,
        flight_energy_per_req=flight_series,
        nonflight_energy_per_req=nonflight_series,
        avg_time=avg_time_series,
        avg_travel=avg_travel_series,
        encodes=encode_series,
        decodes=decode_series,
        semantic_quality=semantic_quality_series,
        semantic_payload_ratio=semantic_payload_ratio_series,
        sinr_samples=sinr_samples,
        summary=summary,
    )


def aggregate(results: Sequence[SimulationResult]) -> SimulationResult:
    attrs = [
        "finished",
        "flight_energy_per_req",
        "nonflight_energy_per_req",
        "avg_time",
        "avg_travel",
        "encodes",
        "decodes",
        "semantic_quality",
        "semantic_payload_ratio",
    ]
    arrays = {name: np.mean([getattr(r, name) for r in results], axis=0) for name in attrs}
    samples: List[float] = []
    for result in results:
        samples.extend(result.sinr_samples)
    summary: Dict[str, float] = {}
    for key in results[0].summary:
        values = [r.summary[key] for r in results]
        summary[key] = float(np.nanmean(values))
    return SimulationResult(
        finished=arrays["finished"],
        flight_energy_per_req=arrays["flight_energy_per_req"],
        nonflight_energy_per_req=arrays["nonflight_energy_per_req"],
        avg_time=arrays["avg_time"],
        avg_travel=arrays["avg_travel"],
        encodes=arrays["encodes"],
        decodes=arrays["decodes"],
        semantic_quality=arrays["semantic_quality"],
        semantic_payload_ratio=arrays["semantic_payload_ratio"],
        sinr_samples=samples,
        summary=summary,
    )


def run_experiments(
    paper: PaperParams,
    assumed: AssumedParams,
    repeats: Optional[int] = None,
    t_slots: Optional[int] = None,
) -> Dict[str, Dict[int, Dict[str, SimulationResult]]]:
    if repeats is not None or t_slots is not None:
        paper = PaperParams(**{**paper.__dict__, "repeats": repeats or paper.repeats, "t_slots": t_slots or paper.t_slots})
    all_results: Dict[str, Dict[int, Dict[str, SimulationResult]]] = {"semantic": {}, "nonsemantic": {}}
    for area in AREAS:
        all_results["semantic"][area] = {}
        for policy in POLICIES:
            reps = [run_single(area, policy, repeat, True, paper, assumed) for repeat in range(paper.repeats)]
            all_results["semantic"][area][policy] = aggregate(reps)
    baseline_area = 300
    all_results["nonsemantic"][baseline_area] = {}
    for policy in OPT_POLICIES:
        reps = [run_single(baseline_area, policy, repeat, False, paper, assumed) for repeat in range(paper.repeats)]
        all_results["nonsemantic"][baseline_area][policy] = aggregate(reps)
    return all_results


def draw_line_panel(
    draw: ImageDraw.ImageDraw,
    box: Tuple[int, int, int, int],
    series: Dict[str, Tuple[np.ndarray, np.ndarray]],
    title: str,
    y_label: str,
    colors: Dict[str, Tuple[int, int, int]],
    y_limits: Optional[Tuple[float, float]] = None,
    x_limits: Optional[Tuple[float, float]] = None,
) -> None:
    left, top, right, bottom = box
    title_font = _font(15, True)
    label_font = _font(11)
    tick_font = _font(10)
    plot_left = left + 52
    plot_top = top + 30
    plot_right = right - 16
    plot_bottom = bottom - 42
    draw.text((left + 8, top + 6), title, font=title_font, fill=(20, 20, 20))
    draw.rectangle((plot_left, plot_top, plot_right, plot_bottom), outline=(80, 80, 80), width=1)

    xs = np.concatenate([v[0] for v in series.values() if len(v[0])])
    ys = np.concatenate([v[1] for v in series.values() if len(v[1])])
    if len(xs) == 0 or len(ys) == 0:
        return
    xmin, xmax = x_limits if x_limits else (float(np.nanmin(xs)), float(np.nanmax(xs)))
    ymin, ymax = y_limits if y_limits else (float(np.nanmin(ys)), float(np.nanmax(ys)))
    if abs(ymax - ymin) < 1e-12:
        ymax = ymin + 1.0
    ypad = 0.08 * (ymax - ymin)
    if y_limits is None:
        ymin -= ypad
        ymax += ypad

    for frac in (0.25, 0.5, 0.75):
        y = int(plot_bottom - frac * (plot_bottom - plot_top))
        draw.line((plot_left, y, plot_right, y), fill=(225, 225, 225), width=1)
    for frac in (0.25, 0.5, 0.75):
        x = int(plot_left + frac * (plot_right - plot_left))
        draw.line((x, plot_top, x, plot_bottom), fill=(235, 235, 235), width=1)

    def xy(x: float, y: float) -> Tuple[int, int]:
        px = plot_left + (x - xmin) / (xmax - xmin) * (plot_right - plot_left)
        py = plot_bottom - (y - ymin) / (ymax - ymin) * (plot_bottom - plot_top)
        return int(px), int(py)

    for name, (x_values, y_values) in series.items():
        if len(x_values) < 2:
            continue
        pts = [xy(float(x), float(y)) for x, y in zip(x_values, y_values)]
        draw.line(pts, fill=colors.get(name, (0, 0, 0)), width=2)

    for frac in (0.0, 0.5, 1.0):
        x_value = xmin + frac * (xmax - xmin)
        px = int(plot_left + frac * (plot_right - plot_left))
        draw.text((px - 16, plot_bottom + 6), f"{x_value:.0f}", font=tick_font, fill=(40, 40, 40))
    for frac in (0.0, 0.5, 1.0):
        y_value = ymin + frac * (ymax - ymin)
        py = int(plot_bottom - frac * (plot_bottom - plot_top))
        draw.text((left + 4, py - 7), f"{y_value:.1f}", font=tick_font, fill=(40, 40, 40))

    draw.text((plot_left + 90, bottom - 24), "Simulation time (s)" if xmax > 100 else "SINR (dB)", font=label_font, fill=(20, 20, 20))
    draw.text((left + 6, top + 32), y_label, font=label_font, fill=(20, 20, 20))


def save_grid_plot(
    path: Path,
    panels: Sequence[Tuple[str, str, Dict[str, Tuple[np.ndarray, np.ndarray]], Optional[Tuple[float, float]], Optional[Tuple[float, float]]]],
    legend_items: Sequence[str],
    colors: Dict[str, Tuple[int, int, int]],
    cols: int = 2,
    panel_size: Tuple[int, int] = (520, 360),
) -> None:
    rows = math.ceil(len(panels) / cols)
    legend_height = 46
    image = Image.new("RGB", (cols * panel_size[0], rows * panel_size[1] + legend_height), "white")
    draw = ImageDraw.Draw(image)
    for idx, (title, y_label, series, y_lim, x_lim) in enumerate(panels):
        row = idx // cols
        col = idx % cols
        left = col * panel_size[0]
        top = row * panel_size[1]
        draw_line_panel(draw, (left, top, left + panel_size[0], top + panel_size[1]), series, title, y_label, colors, y_lim, x_lim)
    legend_font = _font(13)
    x = 20
    y = rows * panel_size[1] + 12
    for item in legend_items:
        draw.line((x, y + 8, x + 26, y + 8), fill=colors.get(item, (0, 0, 0)), width=3)
        draw.text((x + 32, y), item, font=legend_font, fill=(20, 20, 20))
        x += 132
    image.save(path)


def make_plots(results: Dict[str, Dict[int, Dict[str, SimulationResult]]], out_dir: Path, paper: PaperParams) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    colors = {
        "Stochastic": (65, 105, 225),
        "LinUCB": (25, 130, 95),
        "SA": (220, 130, 35),
        "Greedy": (185, 55, 65),
        "MCTS": (120, 75, 170),
        "LinUCB_ns": (25, 130, 95),
        "SA_ns": (220, 130, 35),
        "Greedy_ns": (185, 55, 65),
        "MCTS_ns": (120, 75, 170),
    }
    t = np.arange(paper.t_slots + 1)
    paths: Dict[str, Path] = {}

    def area_panels(metric: str, y_label: str, policies: Sequence[str], y_limits: Optional[Tuple[float, float]] = None) -> List:
        panels = []
        for area in AREAS:
            series = {policy: (t, getattr(results["semantic"][area][policy], metric)) for policy in policies}
            panels.append((f"{area} x {area} m2", y_label, series, y_limits, (0, paper.t_slots)))
        return panels

    plots = [
        ("finished_requests.png", "finished", "Finished requests", POLICIES, None),
        ("flight_energy_per_request.png", "flight_energy_per_req", "J / request", OPT_POLICIES, None),
        ("nonflight_energy_per_request.png", "nonflight_energy_per_req", "J / request", OPT_POLICIES, None),
        ("average_time_cost.png", "avg_time", "Seconds", OPT_POLICIES, None),
        ("average_travel_distance.png", "avg_travel", "Meters", POLICIES, None),
        ("encode_counts.png", "encodes", "Encode count", POLICIES, None),
        ("decode_counts.png", "decodes", "Decode count", POLICIES, None),
    ]
    for filename, metric, y_label, policies, y_limits in plots:
        path = out_dir / filename
        save_grid_plot(path, area_panels(metric, y_label, policies, y_limits), policies, colors, cols=2)
        paths[filename] = path

    cdf_panels = []
    for area in AREAS:
        series = {}
        for policy in POLICIES:
            samples = np.array(results["semantic"][area][policy].sinr_samples, dtype=float)
            if len(samples) == 0:
                xs = np.array([0.0])
                ys = np.array([0.0])
            else:
                xs = np.sort(samples)
                ys = np.arange(1, len(xs) + 1) / len(xs)
            series[policy] = (xs, ys)
        cdf_panels.append((f"{area} x {area} m2", "CDF", series, (0.0, 1.0), (-40.0, 60.0)))
    path = out_dir / "sinr_cdf.png"
    save_grid_plot(path, cdf_panels, POLICIES, colors, cols=2)
    paths["sinr_cdf.png"] = path

    baseline_panels = []
    baseline_area = 300
    baseline_map = [
        ("finished", "Finished requests", "Number of finished requests", None),
        ("flight_energy_per_req", "J / request", "Flight energy per request", None),
        ("nonflight_energy_per_req", "J / request", "Non-flight energy per request", None),
        ("avg_time", "Seconds", "Average time cost", None),
        ("avg_travel", "Meters", "Average travel distance", None),
    ]
    for metric, y_label, title, y_lim in baseline_map:
        series = {}
        for policy in OPT_POLICIES:
            series[policy] = (t, getattr(results["semantic"][baseline_area][policy], metric))
            series[f"{policy}_ns"] = (t, getattr(results["nonsemantic"][baseline_area][policy], metric))
        baseline_panels.append((title, y_label, series, y_lim, (0, paper.t_slots)))
    series = {}
    for policy in OPT_POLICIES:
        samples = np.sort(np.array(results["semantic"][baseline_area][policy].sinr_samples, dtype=float))
        series[policy] = (samples, np.arange(1, len(samples) + 1) / max(len(samples), 1))
        samples_ns = np.sort(np.array(results["nonsemantic"][baseline_area][policy].sinr_samples, dtype=float))
        series[f"{policy}_ns"] = (samples_ns, np.arange(1, len(samples_ns) + 1) / max(len(samples_ns), 1))
    baseline_panels.append(("CDF of SINR", "CDF", series, (0.0, 1.0), (-40.0, 60.0)))
    ns_colors = dict(colors)
    for policy in OPT_POLICIES:
        ns_colors[f"{policy}_ns"] = tuple(max(0, int(c * 0.55)) for c in colors[policy])
    path = out_dir / "semantic_vs_nonsemantic_300m.png"
    save_grid_plot(path, baseline_panels, list(OPT_POLICIES) + [f"{p}_ns" for p in OPT_POLICIES], ns_colors, cols=2)
    paths["semantic_vs_nonsemantic_300m.png"] = path
    return paths


def write_summary_csv(results: Dict[str, Dict[int, Dict[str, SimulationResult]]], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "summary_metrics.csv"
    core_fields = [
        "mode",
        "area",
        "policy",
        "finished",
        "flight_energy_per_req",
        "nonflight_energy_per_req",
        "avg_time",
        "avg_travel",
        "encodes",
        "decodes",
        "semantic_quality",
        "semantic_payload_ratio",
        "sinr_median_db",
    ]
    extra_fields = sorted(
        {
            key
            for mode_results in results.values()
            for area_results in mode_results.values()
            for result in area_results.values()
            for key in result.summary
            if key not in core_fields
        }
    )
    fields = core_fields + extra_fields
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for mode, mode_results in results.items():
            for area, area_results in mode_results.items():
                for policy, result in area_results.items():
                    row = {field: "" for field in fields}
                    row.update({"mode": mode, "area": area, "policy": policy})
                    row.update(result.summary)
                    writer.writerow(row)
    return path


def write_timeseries_npz(results: Dict[str, Dict[int, Dict[str, SimulationResult]]], out_dir: Path) -> Path:
    arrays = {}
    for mode, mode_results in results.items():
        for area, area_results in mode_results.items():
            for policy, result in area_results.items():
                prefix = f"{mode}_{area}_{policy}"
                arrays[f"{prefix}_finished"] = result.finished
                arrays[f"{prefix}_flight"] = result.flight_energy_per_req
                arrays[f"{prefix}_nonflight"] = result.nonflight_energy_per_req
                arrays[f"{prefix}_avg_time"] = result.avg_time
                arrays[f"{prefix}_avg_travel"] = result.avg_travel
                arrays[f"{prefix}_encodes"] = result.encodes
                arrays[f"{prefix}_decodes"] = result.decodes
                arrays[f"{prefix}_semantic_quality"] = result.semantic_quality
                arrays[f"{prefix}_semantic_payload_ratio"] = result.semantic_payload_ratio
                arrays[f"{prefix}_sinr"] = np.array(result.sinr_samples, dtype=float)
    path = out_dir / "timeseries_and_sinr_samples.npz"
    np.savez_compressed(path, **arrays)
    return path


def write_run_metadata(
    out_dir: Path,
    paper: PaperParams,
    assumed: AssumedParams,
    elapsed_seconds: float,
    semantic_profile: Optional[Dict[str, object]] = None,
) -> Path:
    path = out_dir / "run_metadata.json"
    payload = {
        "paper_params": paper.__dict__,
        "assumed_params": assumed.__dict__,
        "semantic_profile": semantic_profile or {"source": "paper_table_iii", "applied": False},
        "elapsed_seconds": elapsed_seconds,
        "notes": [
            "Cityscapes access requires login at the official dataset site. If semantic_profile.applied is true, the run uses the measured substitute dataset profile; otherwise it uses paper Table III values.",
            "No official AirTalking source code was found by title, DOI, or algorithm-name search during this reproduction.",
            "Unspecified simulator constants are isolated under AssumedParams.",
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce AirTalking UAV semantic D2D simulations from paper equations.")
    parser.add_argument("--out", default="outputs/airtalking_reproduction", help="Output directory.")
    parser.add_argument("--repeats", type=int, default=None, help="Override repeat count for quick tests.")
    parser.add_argument("--t-slots", type=int, default=None, help="Override T for quick tests.")
    parser.add_argument("--semantic-summary", default=None, help="JSON summary from measure_camvid_semantics.py.")
    parser.add_argument("--semantic-raw-basis", choices=["uncompressed", "png"], default="uncompressed")
    parser.add_argument("--semantic-profile-kind", choices=["zlib", "feature"], default="zlib")
    parser.add_argument("--semantic-encoder-mode", choices=["measured", "paper"], default="measured")
    parser.add_argument("--semantic-decoder-mode", choices=["measured", "paper"], default="measured")
    parser.add_argument(
        "--assumed",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override an AssumedParams field, e.g. --assumed workload_mean_bits=80000000.",
    )
    args = parser.parse_args()

    import time

    paper = PaperParams()
    assumed = apply_assumed_overrides(AssumedParams(), args.assumed)
    if args.repeats is not None or args.t_slots is not None:
        paper = PaperParams(**{**paper.__dict__, "repeats": args.repeats or paper.repeats, "t_slots": args.t_slots or paper.t_slots})
    semantic_summary = Path(args.semantic_summary) if args.semantic_summary else None
    paper, semantic_profile = apply_semantic_summary(
        paper,
        semantic_summary,
        args.semantic_raw_basis,
        args.semantic_encoder_mode,
        args.semantic_decoder_mode,
        args.semantic_profile_kind,
    )
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    results = run_experiments(paper, assumed)
    elapsed = time.perf_counter() - started
    csv_path = write_summary_csv(results, out_dir)
    npz_path = write_timeseries_npz(results, out_dir)
    metadata_path = write_run_metadata(out_dir, paper, assumed, elapsed, semantic_profile)
    plot_paths = make_plots(results, out_dir / "figures", paper)
    print(json.dumps(
        {
            "out_dir": str(out_dir),
            "summary_csv": str(csv_path),
            "npz": str(npz_path),
            "metadata": str(metadata_path),
            "figures": {name: str(path) for name, path in plot_paths.items()},
            "elapsed_seconds": elapsed,
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
