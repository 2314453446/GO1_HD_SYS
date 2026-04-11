#!/usr/bin/env python3
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Tuple
import numpy as np


class RowEndState(Enum):
    FOLLOWING = auto()
    APPROACHING = auto()
    REACHED = auto()


@dataclass
class RowEndParams:
    cross_detect_zone_ratio: float = 0.20
    cross_heat_row_threshold: float = 0.30
    cross_min_hot_ratio: float = 0.45
    furrow_vanish_threshold: float = 0.10
    cross_confirm_frames: int = 5
    reach_distance_m: float = 0.25
    cross_u_smooth: float = 0.7
    cross_v_smooth: float = 0.7
    center_band_half_width: int = 40


def detect_row_end(
    H: np.ndarray,
    vis_mask: np.ndarray,
    u_ref: float,
    v0: int, v1: int, u0: int, u1: int,
    params: RowEndParams,
) -> Tuple[bool, Optional[float], Optional[float], dict]:
    h_roi = max(1, v1 - v0)
    detect_rows_count = max(3, int(h_roi * params.cross_detect_zone_ratio))
    detect_v_start = max(v0, v1 - detect_rows_count)
    detect_v_end = v1

    hot_rows = []
    cross_u_candidates = []
    row_means = []

    for vr in range(detect_v_start, detect_v_end):
        row_heat = H[vr, u0:u1]
        row_mask = vis_mask[vr, u0:u1] > 0
        if not np.any(row_mask):
            continue

        valid_heat = row_heat[row_mask]
        mean_heat = float(np.mean(valid_heat))
        row_means.append(mean_heat)

        if mean_heat >= params.cross_heat_row_threshold:
            hot_rows.append(vr)
            cols = np.where(row_mask)[0] + u0
            weights = row_heat[row_mask].astype(np.float64)
            ws = float(np.sum(weights))
            if ws > 1e-9:
                weighted_u = float(np.sum(cols * weights) / ws)
            else:
                weighted_u = float(u_ref)
            cross_u_candidates.append(weighted_u)

    total_rows = max(1, detect_v_end - detect_v_start)
    hot_ratio = len(hot_rows) / total_rows

    ul = max(u0, int(round(u_ref - params.center_band_half_width)))
    ur = min(u1, int(round(u_ref + params.center_band_half_width)))

    center_band_vals = []
    for vr in range(detect_v_start, detect_v_end):
        band = H[vr, ul:ur]
        band_mask = vis_mask[vr, ul:ur] > 0
        if np.any(band_mask):
            center_band_vals.append(float(np.mean(band[band_mask])))

    mean_center_heat = float(np.mean(center_band_vals)) if center_band_vals else 0.0

    detected = (
        hot_ratio >= params.cross_min_hot_ratio
        and mean_center_heat < params.furrow_vanish_threshold
    )

    if detected:
        cross_v = float(np.mean(hot_rows)) if hot_rows else float(0.5 * (detect_v_start + detect_v_end))
        cross_u = float(np.mean(cross_u_candidates)) if cross_u_candidates else float(u_ref)
    else:
        cross_v = None
        cross_u = None

    debug = {
        "detect_v_start": detect_v_start,
        "detect_v_end": detect_v_end,
        "hot_ratio": hot_ratio,
        "mean_center_heat": mean_center_heat,
        "num_hot_rows": len(hot_rows),
    }

    return detected, cross_v, cross_u, debug


class RowEndManager:
    def __init__(self, params: Optional[RowEndParams] = None):
        self.params = params if params is not None else RowEndParams()
        self.state = RowEndState.FOLLOWING
        self._confirm_count = 0
        self._cross_u_smooth: Optional[float] = None
        self._cross_v_smooth: Optional[float] = None
        self._cross_3d: Optional[np.ndarray] = None
        self._debug = {}

    def reset(self):
        self.state = RowEndState.FOLLOWING
        self._confirm_count = 0
        self._cross_u_smooth = None
        self._cross_v_smooth = None
        self._cross_3d = None
        self._debug = {}

    def update(
        self,
        H: np.ndarray,
        vis_mask: np.ndarray,
        u_ref: float,
        v0: int, v1: int, u0: int, u1: int,
        D: np.ndarray,
        K,
        R: np.ndarray,
        t: np.ndarray,
        current_pos_3d: Optional[np.ndarray] = None,
    ) -> RowEndState:
        if self.state == RowEndState.REACHED:
            return self.state

        detected, cross_v, cross_u, debug = detect_row_end(
            H, vis_mask, u_ref, v0, v1, u0, u1, self.params
        )
        self._debug = debug

        if detected:
            self._confirm_count += 1
            au = self.params.cross_u_smooth
            av = self.params.cross_v_smooth

            if self._cross_u_smooth is None:
                self._cross_u_smooth = cross_u
                self._cross_v_smooth = cross_v
            else:
                self._cross_u_smooth = au * self._cross_u_smooth + (1.0 - au) * cross_u
                self._cross_v_smooth = av * self._cross_v_smooth + (1.0 - av) * cross_v
        else:
            if self.state == RowEndState.FOLLOWING:
                self._confirm_count = max(0, self._confirm_count - 1)

        if self._cross_u_smooth is not None and self._cross_v_smooth is not None:
            ui = int(np.clip(round(self._cross_u_smooth), 0, D.shape[1] - 1))
            vi = int(np.clip(round(self._cross_v_smooth), 0, D.shape[0] - 1))
            z = float(D[vi, ui])

            if np.isfinite(z) and z > 0.0:
                x = (ui - K.cx) * z / K.fx
                y = (vi - K.cy) * z / K.fy
                p = np.array([x, y, z], dtype=np.float64)
                self._cross_3d = R @ p + t

        if self.state == RowEndState.FOLLOWING:
            if self._confirm_count >= self.params.cross_confirm_frames:
                self.state = RowEndState.APPROACHING
        elif self.state == RowEndState.APPROACHING:
            if current_pos_3d is not None and self._cross_3d is not None:
                dist = float(np.linalg.norm(current_pos_3d - self._cross_3d))
                if dist <= self.params.reach_distance_m:
                    self.state = RowEndState.REACHED

        return self.state

    @property
    def cross_point_3d(self) -> Optional[np.ndarray]:
        return None if self._cross_3d is None else self._cross_3d.copy()

    @property
    def cross_uv(self) -> Tuple[Optional[float], Optional[float]]:
        return self._cross_u_smooth, self._cross_v_smooth

    @property
    def debug_info(self) -> dict:
        out = dict(self._debug)
        out["confirm_count"] = self._confirm_count
        out["state"] = self.state.name
        return out