"""Streamlit UI: wide CSV → resistance vs time and IV (current vs voltage) with per-series visibility."""

from __future__ import annotations

import io
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from math import comb
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import qualitative
from plotly.subplots import make_subplots
import streamlit as st

# Crossbar: ``G3:0(S)`` (conductance / state) or ``I3:0(A)`` / ``I3-0(A)`` (per-cell current) → row 3, col 0.
# Same 16×16 layout: top header = row index, left labels = column index (0-based).
GRID_SIZE = 16
_CROSSBAR_RE = re.compile(r"^(?:G|I)(\d+)[:\-](\d+)(?:\([^)]*\))?$", re.IGNORECASE)


def _decode_csv_text(raw: bytes) -> str:
    for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def cleanup_df(df: pd.DataFrame) -> pd.DataFrame:
    """Strip headers; drop empty trailing columns (common with ';' at end of line)."""
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    out = out.dropna(axis=1, how="all")
    empty_named = [c for c in out.columns if c == ""]
    if empty_named:
        out = out.drop(columns=[c for c in empty_named if out[c].isna().all()], errors="ignore")
    return out


def parse_crossbar_cell(column_name: str) -> tuple[int, int] | None:
    """Parse ``G<r>:<c>(…)`` or ``I<r>:<c>(…)`` / hyphen form; return ``(row, col)`` if inside the grid."""
    m = _CROSSBAR_RE.match(str(column_name).strip())
    if not m:
        return None
    r, c = int(m.group(1)), int(m.group(2))
    if 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE:
        return (r, c)
    return None


def render_crossbar_checkbox_grid(grid_map: dict[tuple[int, int], str], *, wp: str, ns: str) -> None:
    """16×16 matrix of checkboxes for cells present in ``grid_map`` (Streamlit keys prefixed by ``wp``)."""
    st.subheader("Crossbar (16×16)")
    st.caption(
        "Check cells to plot. **X (top) = row index**, **Y (left) = column index** "
        "from `G<row>:<col>`, `I<row>:<col>`, or hyphen forms. Hover a checkbox for the CSV column name."
    )
    header = st.columns([0.55] + [1] * GRID_SIZE)
    header[0].write("")
    for r in range(GRID_SIZE):
        header[r + 1].caption(str(r))
    for c in range(GRID_SIZE):
        row_cols = st.columns([0.55] + [1] * GRID_SIZE)
        row_cols[0].markdown(f"**{c}**")
        for r in range(GRID_SIZE):
            cell = (r, c)
            with row_cols[r + 1]:
                if cell in grid_map:
                    cx_key = f"{wp}_cx_{ns}_{r}_{c}"
                    if cx_key not in st.session_state:
                        st.session_state[cx_key] = True
                    st.checkbox(
                        " ",
                        key=cx_key,
                        label_visibility="collapsed",
                        help=grid_map[cell],
                    )
                else:
                    st.write("·")


def crossbar_column_map(y_columns: list[str]) -> dict[tuple[int, int], str]:
    """Map ``(row, col)`` → CSV column name. First column wins if two claim the same cell."""
    d: dict[tuple[int, int], str] = {}
    for col in y_columns:
        cell = parse_crossbar_cell(col)
        if cell is None:
            continue
        d.setdefault(cell, col)
    return d


def _fraction_numeric_cells(df: pd.DataFrame, *, max_cols: int = 8, max_rows: int = 100) -> float:
    """Share of non-empty cells that are already numeric (after read_csv decimal=…)."""
    if df.empty or len(df.columns) < 1:
        return 0.0
    total = 0
    numeric = 0
    for col in list(df.columns)[:max_cols]:
        for v in df[col].head(max_rows):
            if v is None or (isinstance(v, float) and pd.isna(v)):
                continue
            if isinstance(v, str) and not str(v).strip():
                continue
            total += 1
            if isinstance(v, (int, float)):
                numeric += 1
    return numeric / total if total else 0.0


def read_csv_bytes(raw: bytes) -> pd.DataFrame:
    """
    Load instrument-style CSV: comma or semicolon separated, scientific notation (e.g. 1.76E-06).

    Supports European decimal comma (e.g. ``0,1`` and ``1,5789465E-07``) when ``;`` separates fields.
    Picks delimiter and decimal by column count and how many values parse as numbers.
    """
    text = _decode_csv_text(raw)
    # (sep, decimal_char) — skip (",", ","): delimiter and decimal would both be comma.
    parse_attempts: list[tuple[str | None, str]] = [
        (None, "."),
        (None, ","),
        (";", "."),
        (";", ","),
        (",", "."),
    ]
    best: pd.DataFrame | None = None
    best_key = (-1, -1.0)  # (num_columns, numeric_fraction)
    last_exc: Exception | None = None

    for sep, decimal in parse_attempts:
        kw: dict[str, Any] = {"engine": "python", "decimal": decimal}
        if sep is not None:
            kw["sep"] = sep
        else:
            kw["sep"] = None
        try:
            df = pd.read_csv(io.StringIO(text), **kw)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            continue
        df = cleanup_df(df)
        n = len(df.columns)
        if n < 1:
            continue
        key = (n, _fraction_numeric_cells(df))
        if key > best_key:
            best = df
            best_key = key

    if best is None or best_key[0] < 1:
        raise ValueError(last_exc or "Could not parse CSV")
    return best


def infer_x_column(columns: list[str]) -> str:
    if not columns:
        raise ValueError("CSV has no columns")
    preferred_exact = ("time", "timestamp", "date", "t")
    lower_map = {c.lower().strip(): c for c in columns}
    for p in preferred_exact:
        if p in lower_map:
            return lower_map[p]
    # e.g. "Time(s)", "time [s]"
    prefix_starts = ("time", "timestamp", "date")
    for col in columns:
        cl = col.lower().strip()
        if any(cl == ps or cl.startswith(f"{ps}(") or cl.startswith(f"{ps}[") for ps in prefix_starts):
            return col
    return columns[0]


def infer_voltage_column(columns: list[str]) -> str:
    """Pick a likely voltage / bias column for IV sweeps; otherwise first column."""
    if not columns:
        raise ValueError("CSV has no columns")
    preferred_exact = ("voltage", "bias", "v", "volt", "u")
    lower_map = {c.lower().strip(): c for c in columns}
    for p in preferred_exact:
        if p in lower_map:
            return lower_map[p]
    prefix_starts = ("voltage", "bias", "volt", "u")
    for col in columns:
        cl = col.lower().strip()
        if cl == "u" or cl.startswith("u(") or cl.startswith("u["):
            return col
        if cl == "v" or cl.startswith("v(") or cl.startswith("v["):
            return col
        if any(cl == ps or cl.startswith(f"{ps}(") or cl.startswith(f"{ps}[") for ps in prefix_starts):
            return col
    return columns[0]


def numeric_y_columns(df: pd.DataFrame, x_col: str) -> list[str]:
    """Columns plottable as Y: not X, and numeric or convertible with some valid values."""
    out: list[str] = []
    for c in df.columns:
        if c == x_col:
            continue
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            out.append(c)
            continue
        conv = pd.to_numeric(s, errors="coerce")
        if conv.notna().any():
            out.append(c)
    return out


def relative_to_first_valid_percent(series: pd.Series[Any]) -> pd.Series[Any]:
    """Normalize a trace to its first valid point (100%)."""
    num = pd.to_numeric(series, errors="coerce")
    first = num.first_valid_index()
    if first is None:
        return num
    ref = float(num.loc[first])
    if ref == 0:
        # Relative normalization is undefined for zero baseline.
        return pd.Series(float("nan"), index=num.index)
    return (num / ref) * 100.0


def first_valid_numeric_value(series: pd.Series[Any]) -> float | None:
    """Return first numeric value in ``series``; ``None`` when unavailable."""
    num = pd.to_numeric(series, errors="coerce")
    first = num.first_valid_index()
    if first is None:
        return None
    return float(num.loc[first])


def last_valid_numeric_value(series: pd.Series[Any]) -> float | None:
    """Return last numeric value in ``series``; ``None`` when unavailable."""
    num = pd.to_numeric(series, errors="coerce")
    last = num.last_valid_index()
    if last is None:
        return None
    return float(num.loc[last])


def percent_difference(a: float, b: float) -> float:
    """Symmetric percent difference between two start values."""
    denom = max(abs(a), abs(b), 1e-15)
    return abs(a - b) / denom * 100.0


def close_value_groups(
    values: dict[str, float],
    *,
    threshold_percent: float,
) -> dict[str, int]:
    """
    Group named scalar values by closeness.

    Implementation: sort by value and split into a new group when the adjacent percent jump
    exceeds the threshold (see ``percent_difference``).
    """
    items = [(name, value) for name, value in values.items() if value is not None]
    if not items:
        return {}

    items.sort(key=lambda it: it[1])
    group_map: dict[str, int] = {}
    group_id = 0
    prev = items[0][1]
    group_map[items[0][0]] = group_id

    for name, value in items[1:]:
        if percent_difference(prev, value) > threshold_percent:
            group_id += 1
        group_map[name] = group_id
        prev = value
    return group_map


def close_start_groups(
    df: pd.DataFrame,
    traces: list[str],
    *,
    threshold_percent: float,
) -> dict[str, int]:
    """
    Group traces with close first points.

    Implementation: sort by first valid value and split when adjacent percent jump exceeds threshold.
    """
    values: dict[str, float] = {}
    for name in traces:
        v = first_valid_numeric_value(df[name])
        if v is None:
            continue
        values[name] = v
    return close_value_groups(values, threshold_percent=threshold_percent)


def _interp_current_at(voltage: np.ndarray, current: np.ndarray, read_voltage: float) -> float | None:
    """Current at ``read_voltage`` along a branch via linear interpolation (nearest point if out of range)."""
    if voltage.size == 0:
        return None
    order = np.argsort(voltage)
    v_sorted = voltage[order]
    i_sorted = current[order]
    if read_voltage < float(v_sorted[0]) or read_voltage > float(v_sorted[-1]):
        nearest = int(np.argmin(np.abs(voltage - read_voltage)))
        return float(current[nearest])
    return float(np.interp(read_voltage, v_sorted, i_sorted))


def iv_read_resistance(
    voltage: Any,
    current: Any,
    *,
    polarity: str,
    read_voltage: float,
) -> float | None:
    """
    Resistance read from the SET *return* branch at ``read_voltage``.

    A sweep has a rising part (0 → ±Vmax) and a decreasing/return part (±Vmax → 0). The read is taken
    on the **decreasing** branch within the chosen polarity region (the rising part is ignored):

    - ``polarity == "positive"``: branch from the peak (max) voltage back toward 0, restricted to ``v >= 0``.
    - ``polarity == "negative"``: branch from the trough (min) voltage back toward 0, restricted to ``v <= 0``.

    Returns ``R = read_voltage / I(read_voltage)``; ``None`` when there is no usable branch point or ``I == 0``.
    """
    v = pd.to_numeric(pd.Series(voltage), errors="coerce").to_numpy(dtype=float)
    i = pd.to_numeric(pd.Series(current), errors="coerce").to_numpy(dtype=float)
    n = min(v.size, i.size)
    v, i = v[:n], i[:n]
    mask = ~(np.isnan(v) | np.isnan(i))
    v, i = v[mask], i[mask]
    if v.size < 2:
        return None

    if polarity == "positive":
        peak_idx = int(np.argmax(v))
        v_branch, i_branch = v[peak_idx:], i[peak_idx:]
        region = v_branch >= 0
    else:
        peak_idx = int(np.argmin(v))
        v_branch, i_branch = v[peak_idx:], i[peak_idx:]
        region = v_branch <= 0

    v_branch, i_branch = v_branch[region], i_branch[region]
    if v_branch.size < 1:
        return None

    cur = _interp_current_at(v_branch, i_branch, read_voltage)
    if cur is None or cur == 0:
        return None
    return read_voltage / cur


def _sanitize_iv_curve(voltage: Any, current: Any) -> tuple[np.ndarray, np.ndarray] | None:
    """Return sorted finite ``(V, I)`` arrays with duplicate voltages averaged."""
    v = pd.to_numeric(pd.Series(voltage), errors="coerce").to_numpy(dtype=float)
    i = pd.to_numeric(pd.Series(current), errors="coerce").to_numpy(dtype=float)
    n = min(v.size, i.size)
    if n < 2:
        return None
    v, i = v[:n], i[:n]
    mask = np.isfinite(v) & np.isfinite(i)
    v, i = v[mask], i[mask]
    if v.size < 2:
        return None

    order = np.argsort(v)
    v_sorted = v[order]
    i_sorted = i[order]
    v_unique, inv = np.unique(v_sorted, return_inverse=True)
    if v_unique.size < 2:
        return None
    i_sum = np.bincount(inv, weights=i_sorted)
    i_count = np.bincount(inv)
    i_avg = i_sum / i_count
    return v_unique.astype(float), i_avg.astype(float)


def iv_curve_distance(
    voltage_a: Any,
    current_a: Any,
    voltage_b: Any,
    current_b: Any,
    *,
    grid_points: int = 400,
    metric: str = "area",
) -> float | None:
    """Pairwise IV distance on overlap voltage range using selected metric."""
    curve_a = _sanitize_iv_curve(voltage_a, current_a)
    curve_b = _sanitize_iv_curve(voltage_b, current_b)
    if curve_a is None or curve_b is None:
        return None
    v_a, i_a = curve_a
    v_b, i_b = curve_b

    v_min = max(float(v_a[0]), float(v_b[0]))
    v_max = min(float(v_a[-1]), float(v_b[-1]))
    if not np.isfinite(v_min) or not np.isfinite(v_max) or v_min >= v_max:
        return None

    points = max(int(grid_points), 2)
    v_grid = np.linspace(v_min, v_max, points, dtype=float)
    i_a_grid = np.interp(v_grid, v_a, i_a)
    i_b_grid = np.interp(v_grid, v_b, i_b)
    y = np.abs(i_a_grid - i_b_grid)
    if metric == "max":
        return float(np.max(y))
    if metric == "sum":
        return float(np.sum(y))
    if metric == "area":
        dx = np.diff(v_grid)
        return float(np.sum((y[:-1] + y[1:]) * 0.5 * dx))
    raise ValueError(f"Unknown IV distance metric: {metric}")


def iv_pairwise_distance_matrix(
    rows: list[dict[str, Any]],
    iv_df: pd.DataFrame,
    voltage_series: Any,
    *,
    grid_points: int = 400,
    metric: str = "area",
) -> tuple[list[str], np.ndarray]:
    """Pairwise IV distances for rows with valid IV curves."""
    valid_rows: list[dict[str, Any]] = []
    for row in rows:
        curve = _sanitize_iv_curve(voltage_series, iv_df[row["iv_col"]])
        if curve is None:
            continue
        valid_rows.append(row)
    devices = [str(row["device"]) for row in valid_rows]
    n = len(devices)
    dist = np.zeros((n, n), dtype=float)
    if n == 0:
        return devices, dist
    for i in range(n):
        for j in range(i + 1, n):
            d = iv_curve_distance(
                voltage_series,
                iv_df[valid_rows[i]["iv_col"]],
                voltage_series,
                iv_df[valid_rows[j]["iv_col"]],
                grid_points=grid_points,
                metric=metric,
            )
            if d is None or not np.isfinite(d):
                d = float("inf")
            dist[i, j] = d
            dist[j, i] = d
    return devices, dist


def groups_from_pairwise_distances(
    devices: list[str],
    distance_matrix: np.ndarray,
    *,
    threshold: float,
) -> dict[str, int]:
    """Single-link threshold grouping via connected components on pairwise distances."""
    n = len(devices)
    if n == 0:
        return {}
    if distance_matrix.shape != (n, n):
        raise ValueError("distance matrix shape must be NxN for provided devices")

    visited = [False] * n
    groups: dict[str, int] = {}
    gid = 0
    for start in range(n):
        if visited[start]:
            continue
        stack = [start]
        visited[start] = True
        while stack:
            idx = stack.pop()
            groups[devices[idx]] = gid
            neighbors = np.where(distance_matrix[idx] <= threshold)[0]
            for nb in neighbors:
                if nb == idx or visited[int(nb)]:
                    continue
                visited[int(nb)] = True
                stack.append(int(nb))
        gid += 1
    return groups


def adjusted_rand_index(labels_a: list[int], labels_b: list[int]) -> float | None:
    """Adjusted Rand Index between two cluster labelings (contingency-table formula); ``None`` if undefined."""
    if len(labels_a) != len(labels_b) or not labels_a:
        return None
    n = len(labels_a)
    contingency: dict[tuple[int, int], int] = {}
    for la, lb in zip(labels_a, labels_b, strict=False):
        contingency[(la, lb)] = contingency.get((la, lb), 0) + 1
    a_counts = Counter(labels_a)
    b_counts = Counter(labels_b)

    sum_comb_cells = sum(comb(v, 2) for v in contingency.values())
    sum_comb_a = sum(comb(v, 2) for v in a_counts.values())
    sum_comb_b = sum(comb(v, 2) for v in b_counts.values())
    total_comb = comb(n, 2)
    if total_comb == 0:
        return None
    expected = sum_comb_a * sum_comb_b / total_comb
    max_index = 0.5 * (sum_comb_a + sum_comb_b)
    if max_index == expected:
        # Both labelings are trivial (single cluster each); treat perfect agreement as 1.0.
        return 1.0
    return (sum_comb_cells - expected) / (max_index - expected)


def is_conductance_column(column_name: str) -> bool:
    """Heuristic: crossbar conductance naming, e.g. ``G3:0(S)``."""
    name = str(column_name).strip()
    if re.match(r"^G\d+[:\-]\d+(?:\([^)]*\))?$", name, re.IGNORECASE):
        return True
    upper = name.upper()
    return "(S)" in upper or upper.endswith("_S")


def _scalar_to_conductance(value: float, column_name: str) -> float:
    """Express a scalar trace value as conductance (S); resistance columns use ``1/R``."""
    return value if is_conductance_column(column_name) else (1.0 / value)


def _retention_conductance_metrics(
    ret_df: pd.DataFrame,
    ret_col: str,
) -> tuple[float | None, float | None, float | None]:
    """Return ``(G_t0, G_final, retention_percent)`` in conductance space for one device."""
    raw_start = first_valid_numeric_value(ret_df[ret_col])
    raw_end = last_valid_numeric_value(ret_df[ret_col])
    if raw_start is None or raw_end is None:
        return None, None, None
    g_t0 = _scalar_to_conductance(raw_start, ret_col)
    g_final = _scalar_to_conductance(raw_end, ret_col)
    if not (np.isfinite(g_t0) and np.isfinite(g_final) and g_t0 > 0):
        return g_t0, g_final, None
    retention = 100.0 * g_final / g_t0
    return g_t0, g_final, retention


def _aligned_finite_pairs(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    mask = np.isfinite(xa) & np.isfinite(ya)
    return xa[mask], ya[mask]


def _pearson(x: np.ndarray, y: np.ndarray) -> tuple[float | None, int]:
    xa, ya = _aligned_finite_pairs(x, y)
    n = int(xa.size)
    if n < 2:
        return None, n
    return float(pd.Series(xa).corr(pd.Series(ya), method="pearson")), n


def _spearman(x: np.ndarray, y: np.ndarray) -> tuple[float | None, int]:
    """Spearman rho via Pearson on average ranks (no scipy dependency)."""
    xa, ya = _aligned_finite_pairs(x, y)
    n = int(xa.size)
    if n < 2:
        return None, n
    rx = pd.Series(xa).rank(method="average").to_numpy(dtype=float)
    ry = pd.Series(ya).rank(method="average").to_numpy(dtype=float)
    return float(pd.Series(rx).corr(pd.Series(ry), method="pearson")), n


def _partial_correlation_pearson(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float | None:
    r_xy, _ = _pearson(x, y)
    r_xz, _ = _pearson(x, z)
    r_yz, _ = _pearson(y, z)
    if r_xy is None or r_xz is None or r_yz is None:
        return None
    denom = (1.0 - r_xz * r_xz) * (1.0 - r_yz * r_yz)
    if denom <= 0:
        return None
    return (r_xy - r_xz * r_yz) / float(np.sqrt(denom))


def _maybe_log10(values: np.ndarray, *, use_log: bool) -> np.ndarray:
    if not use_log:
        return values
    out = values.astype(float, copy=True)
    if np.any(out <= 0):
        return values
    return np.log10(out)


def color_with_alpha(color: str, alpha: float) -> str:
    """Convert hex/rgb color to rgba with requested alpha."""
    c = str(color).strip()
    if c.startswith("#") and len(c) == 7:
        r = int(c[1:3], 16)
        g = int(c[3:5], 16)
        b = int(c[5:7], 16)
        return f"rgba({r},{g},{b},{alpha})"
    if c.startswith("rgb(") and c.endswith(")"):
        inner = c[4:-1]
        return f"rgba({inner},{alpha})"
    return color


def add_group_summary_overlays(
    fig: go.Figure,
    df: pd.DataFrame,
    x_col: str,
    traces: list[str],
    groups: dict[str, int],
    group_color_map: dict[int, str],
    *,
    log_y: bool,
    log_y_use_abs_y: bool,
) -> None:
    """Add group average line and min/max band for plotted groups."""
    x_series, _ = prepare_x_axis(df, x_col)
    group_ids = sorted({groups[t] for t in traces if t in groups})
    for gid in group_ids:
        group_traces = [t for t in traces if groups.get(t) == gid]
        if not group_traces:
            continue
        y_frame = pd.DataFrame(index=df.index)
        for t in group_traces:
            s = pd.to_numeric(df[t], errors="coerce")
            if log_y and log_y_use_abs_y:
                s = s.abs()
            y_frame[t] = s

        y_avg = y_frame.mean(axis=1, skipna=True)
        y_min = y_frame.min(axis=1, skipna=True)
        y_max = y_frame.max(axis=1, skipna=True)
        if log_y:
            y_avg = y_avg.where(y_avg > 0)
            y_min = y_min.where(y_min > 0)
            y_max = y_max.where(y_max > 0)

        base = group_color_map.get(gid, qualitative.Plotly[gid % len(qualitative.Plotly)])
        band = color_with_alpha(base, 0.18)
        avg_line = color_with_alpha(base, 1.0)

        fig.add_trace(
            go.Scatter(
                x=x_series,
                y=y_max,
                mode="lines",
                line=dict(width=0, color=band),
                showlegend=False,
                hoverinfo="skip",
                name=f"Group {gid + 1} max",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x_series,
                y=y_min,
                mode="lines",
                line=dict(width=0, color=band),
                fill="tonexty",
                fillcolor=band,
                showlegend=False,
                hoverinfo="skip",
                name=f"Group {gid + 1} min",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x_series,
                y=y_avg,
                mode="lines",
                line=dict(color=avg_line, width=3),
                name=f"Group {gid + 1} avg",
                connectgaps=False,
            )
        )


def _r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination in the original (untransformed) y space."""
    ss_res = float(np.nansum((y_true - y_pred) ** 2))
    ss_tot = float(np.nansum((y_true - np.nanmean(y_true)) ** 2))
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return 1.0 - ss_res / ss_tot


def fit_retention_form(x: Any, y: Any) -> tuple[str, float, str]:
    """
    Best-fitting simple model of ``y(x)`` among constant/linear/logarithmic/exponential/power.

    Returns ``(name, r_squared, formula)``. R² is always measured in the original y space so the
    candidates are comparable. Log/power are only tried when ``x > 0`` everywhere; exponential/power
    only when ``y > 0`` everywhere.
    """
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    mask = np.isfinite(xa) & np.isfinite(ya)
    xa, ya = xa[mask], ya[mask]
    if xa.size < 3:
        return ("n/a", float("nan"), "too few points")
    order = np.argsort(xa)
    xa, ya = xa[order], ya[order]

    y_mean = float(np.mean(ya))
    spread = float(np.max(ya) - np.min(ya))
    if y_mean != 0 and spread / abs(y_mean) < 0.02:
        return ("constant", 1.0, f"y \u2248 {y_mean:.3g}")

    candidates: list[tuple[str, float, str]] = []
    a, b = np.polyfit(xa, ya, 1)
    candidates.append(("linear", _r_squared(ya, a * xa + b), f"y = {a:.3g}\u00b7x + {b:.3g}"))
    if np.all(xa > 0):
        a, b = np.polyfit(np.log(xa), ya, 1)
        candidates.append(("logarithmic", _r_squared(ya, a * np.log(xa) + b), f"y = {a:.3g}\u00b7ln(x) + {b:.3g}"))
    if np.all(ya > 0):
        b, ln_a = np.polyfit(xa, np.log(ya), 1)
        amp = float(np.exp(ln_a))
        candidates.append(("exponential", _r_squared(ya, amp * np.exp(b * xa)), f"y = {amp:.3g}\u00b7e^({b:.3g}\u00b7x)"))
    if np.all(xa > 0) and np.all(ya > 0):
        b, ln_a = np.polyfit(np.log(xa), np.log(ya), 1)
        amp = float(np.exp(ln_a))
        candidates.append(("power", _r_squared(ya, amp * np.power(xa, b)), f"y = {amp:.3g}\u00b7x^{b:.3g}"))

    return max(candidates, key=lambda c: c[1])


def group_info_table(
    df: pd.DataFrame,
    x_col: str,
    group_to_cols: dict[int, list[str]],
    *,
    value_label: str,
    read_resistances: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Per-group summary: device count, min/max first & last trace values, and best-fit function of the average curve.

    When ``read_resistances`` (column -> IV read resistance in ohms) is given, also reports the group's
    **average read resistance** and **average read conductance**.
    """
    def _fmt(v: float | None) -> str:
        # Scientific notation with 8 significant figures so small values (e.g. ~1e-6 conductance)
        # keep full instrument precision instead of being rounded to "0.000006".
        return "" if v is None else f"{v:.8g}"

    x_num = pd.to_numeric(df[x_col], errors="coerce").to_numpy(dtype=float)
    rows: list[dict[str, Any]] = []
    for gid in sorted(group_to_cols):
        cols = group_to_cols[gid]
        starts = [v for v in (first_valid_numeric_value(df[c]) for c in cols) if v is not None]
        ends = [v for v in (last_valid_numeric_value(df[c]) for c in cols) if v is not None]
        y_frame = pd.DataFrame({c: pd.to_numeric(df[c], errors="coerce") for c in cols})
        y_avg = y_frame.mean(axis=1, skipna=True).to_numpy(dtype=float)
        name, r2, formula = fit_retention_form(x_num, y_avg)
        row: dict[str, Any] = {
            "Group": gid + 1,
            "Devices": len(cols),
            f"Start {value_label} (min)": _fmt(min(starts) if starts else None),
            f"Start {value_label} (max)": _fmt(max(starts) if starts else None),
            f"End {value_label} (min)": _fmt(min(ends) if ends else None),
            f"End {value_label} (max)": _fmt(max(ends) if ends else None),
        }
        if read_resistances is not None:
            r_vals = [read_resistances[c] for c in cols if read_resistances.get(c) is not None]
            g_vals = [1.0 / r for r in r_vals if r != 0]
            row["Avg read R (\u03a9)"] = _fmt(sum(r_vals) / len(r_vals) if r_vals else None)
            row["Avg read G (S)"] = _fmt(sum(g_vals) / len(g_vals) if g_vals else None)
        row["Best fit"] = name
        row["R\u00b2"] = round(r2, 4) if r2 == r2 else None  # noqa: PLR0124 — NaN check
        row["Formula (x = X axis)"] = formula
        rows.append(row)
    return pd.DataFrame(rows)


def render_group_info(
    df: pd.DataFrame,
    x_col: str,
    group_to_cols: dict[int, list[str]],
    *,
    value_label: str,
    key: str,
    read_resistances: dict[str, float] | None = None,
) -> None:
    """Render a per-group information table inside an expander."""
    if not group_to_cols:
        return
    info = group_info_table(df, x_col, group_to_cols, value_label=value_label, read_resistances=read_resistances)
    with st.expander("Group information"):
        read_note = (
            " It also reports the group's **average read resistance / conductance** at the read voltage."
            if read_resistances is not None
            else ""
        )
        st.caption(
            "Per group: number of devices, the min/max of each device's **first** and **last** trace value, "
            "and the **best-fit function** of the group's **average curve** over the X axis (chosen by R\u00b2 among "
            "constant, linear, logarithmic, exponential, power)." + read_note
        )
        st.dataframe(info, use_container_width=True, key=f"group_info_{key}")


def prepare_x_axis(df: pd.DataFrame, x_col: str) -> tuple[pd.Series[Any], str]:
    """Return values for Plotly x and a label for the axis.

    Numeric columns like ``Time(s)`` with values ``0, 900, 1800`` (seconds) must stay **numeric**.
    ``pd.to_datetime`` on integers interprets them as **nanoseconds** since the Unix epoch, so every
    point collapses near 1970 — use a numeric axis instead. Only convert from numeric when values look
    like Unix **seconds** or **milliseconds** (large magnitudes).
    """
    s = df[x_col]
    num = pd.to_numeric(s, errors="coerce")
    frac_num = float(num.notna().mean()) if len(s) else 0.0

    if frac_num >= 0.9:
        vmax = float(num.max())
        # Unix time in ms (e.g. ~1.7e12 for year 2024)
        if vmax > 1e12:
            dt = pd.to_datetime(num, unit="ms", errors="coerce")
            if dt.notna().mean() >= 0.9:
                return dt, "Time"
        # Unix time in seconds (e.g. > 1e9 after ~2001)
        if vmax > 1e9:
            dt = pd.to_datetime(num, unit="s", errors="coerce")
            if dt.notna().mean() >= 0.9:
                return dt, "Time"
        # Elapsed time / index (seconds, minutes, hours in same CSV) — plot as numbers
        return num, x_col

    # Mostly non-numeric strings: try calendar/datetime strings
    dt = pd.to_datetime(s, errors="coerce")
    if len(s) and dt.notna().mean() >= 0.9:
        return dt, "Time"
    return num, x_col


def plot_lines(
    df: pd.DataFrame,
    x_col: str,
    y_cols: list[str],
    *,
    log_y: bool = True,
    y_quantity_label: str = "Resistance",
    log_y_use_abs_y: bool = False,
    trace_color_map: dict[str, str] | None = None,
) -> go.Figure:
    x_series, x_title = prepare_x_axis(df, x_col)
    fig = go.Figure()
    for name in y_cols:
        y_raw = df[name]
        y_series = y_raw if pd.api.types.is_numeric_dtype(y_raw) else pd.to_numeric(y_raw, errors="coerce")
        if log_y and log_y_use_abs_y:
            y_series = y_series.abs()
        fig.add_trace(
            go.Scatter(
                x=x_series,
                y=y_series,
                mode="lines",
                name=name,
                connectgaps=False,
                line=dict(color=trace_color_map[name]) if trace_color_map and name in trace_color_map else None,
            )
        )
    if log_y and log_y_use_abs_y:
        yaxis_title = f"|{y_quantity_label}| (log scale)"
    elif log_y:
        yaxis_title = f"{y_quantity_label} (log scale)"
    else:
        yaxis_title = y_quantity_label
    fig.update_layout(
        margin=dict(l=40, r=24, t=48, b=40),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="right",
            x=0.99,
            # Plotly default is "toggleothers" on double-click; disable that
            # so users don't accidentally isolate a single trace.
            itemdoubleclick="toggle",
        ),
        xaxis_title=x_title,
        yaxis_title=yaxis_title,
        hovermode="x unified",
        height=640,
    )
    if log_y:
        # Power ticks (10ⁿ) and minor grid make log scaling obvious; narrow Y ranges can still
        # look similar to linear because traces only span a small multiplicative band.
        fig.update_yaxes(
            type="log",
            exponentformat="power",
            showexponent="all",
            showgrid=True,
            minor=dict(showgrid=True, gridwidth=1),
        )
    else:
        fig.update_yaxes(type="linear")
    return fig


@dataclass(frozen=True)
class WideCsvViewConfig:
    """Shared settings for retention vs IV wide-table views (sidebar + main layout)."""

    wp: str
    df_key: str
    csv_id_key: str
    x_guess_key: str
    chk_ns_key: str
    infer_default_x: Callable[[list[str]], str]
    uploader_label: str
    uploader_key: str
    empty_info: str
    x_selectbox_label: str
    log_y_checkbox_label: str
    log_y_default: bool
    y_quantity_label: str
    non_positive_log_warning: str
    log_checkbox_help: str | None
    crossbar_example: str
    trace_kind_tip: str
    log_y_use_abs_y: bool = False


RETENTION_VIEW = WideCsvViewConfig(
    wp="ret",
    df_key="retention_df",
    csv_id_key="_retention_csv_id",
    x_guess_key="_retention_x_guess",
    chk_ns_key="_retention_chk_ns",
    infer_default_x=infer_x_column,
    uploader_label="Upload CSV",
    uploader_key="retention_file_uploader",
    empty_info=(
        "Upload a **wide** CSV: one column for time (or index) and **one column per resistance trace**. "
        "Use the **sidebar** to pick the X axis and which series to show. "
        "When columns match `G<row>:<col>(…)` or `I<row>:<col>(…)` (same grid layout), use **Pick devices on 16×16 crossbar** and the grid in the main area."
    ),
    x_selectbox_label="X axis (time)",
    log_y_checkbox_label="Logarithmic Y",
    log_y_default=True,
    y_quantity_label="Y",
    non_positive_log_warning=(
        "Some selected values are **≤ 0**; they cannot be shown on a log axis "
        "and may disappear. Turn off **Logarithmic Y** in the sidebar to see them."
    ),
    log_checkbox_help=None,
    crossbar_example="G3:0(S)",
    trace_kind_tip="resistance",
)

IV_VIEW = WideCsvViewConfig(
    wp="iv",
    df_key="iv_df",
    csv_id_key="_iv_csv_id",
    x_guess_key="_iv_x_guess",
    chk_ns_key="_iv_chk_ns",
    infer_default_x=infer_voltage_column,
    uploader_label="Upload IV CSV",
    uploader_key="iv_file_uploader",
    empty_info=(
        "Upload a **wide** IV CSV: **one voltage column** (e.g. `U(V)`, `Voltage`, `V(V)`) and **one numeric column per "
        "device** (e.g. `I3:0(A)` or `I3-0(A)` — row, then column, top row is row 0 / left column is col 0), one row per sweep point. "
        "Use the **sidebar** like retention: pick the voltage column, enable **Pick devices on 16×16 crossbar** "
        "when `I…` or `G…` columns are present, then select devices on the **main-area grid** below."
    ),
    x_selectbox_label="X axis (voltage)",
    log_y_checkbox_label="Logarithmic current (Y)",
    log_y_default=False,
    y_quantity_label="Current",
    non_positive_log_warning=(
        "Some selected currents are **exactly zero**; log scale cannot display them and they may disappear. "
        "Turn off **Logarithmic current (Y)** to see signed current including zeros."
    ),
    log_checkbox_help="Log Y plots **|current|** so positive and negative branches are visible.",
    crossbar_example="I3:0(A)",
    trace_kind_tip="current",
    log_y_use_abs_y=True,
)


def _x_axis_state_key(wp: str) -> str:
    return f"{wp}_x_axis_column"


def _wide_csv_sidebar_controls(cfg: WideCsvViewConfig) -> None:
    """Upload + axis + series controls. Call only inside ``with st.sidebar:`` (no parentheses)."""
    wp = cfg.wp
    uploaded = st.file_uploader(cfg.uploader_label, type=["csv"], key=cfg.uploader_key)

    if uploaded is None:
        st.info(cfg.empty_info)
        return

    raw = uploaded.getvalue()
    csv_id = (uploaded.name, len(raw))
    if st.session_state.get(cfg.csv_id_key) != csv_id:
        st.session_state[cfg.csv_id_key] = csv_id
        try:
            df = read_csv_bytes(raw)
        except Exception as exc:  # noqa: BLE001 — surface parse errors in UI
            st.error(f"Could not parse CSV: {exc}")
            return
        if df.empty:
            st.warning("The CSV has no rows.")
            return
        st.session_state[cfg.df_key] = df
        st.session_state[cfg.x_guess_key] = cfg.infer_default_x(list(df.columns))
        st.session_state[cfg.chk_ns_key] = f"{uploaded.name}_{len(raw)}"
        st.session_state[_x_axis_state_key(wp)] = st.session_state[cfg.x_guess_key]

    df = st.session_state[cfg.df_key]
    all_cols = list(df.columns)

    chk_ns = str(st.session_state[cfg.chk_ns_key])
    xa_key = _x_axis_state_key(wp)

    default_x = st.session_state[cfg.x_guess_key] if st.session_state[cfg.x_guess_key] in all_cols else all_cols[0]
    cur = st.session_state.get(xa_key, default_x)
    if cur not in all_cols:
        cur = default_x
        st.session_state[xa_key] = cur

    st.selectbox(
        cfg.x_selectbox_label,
        options=all_cols,
        index=all_cols.index(cur),
        key=xa_key,
    )

    st.checkbox(
        cfg.log_y_checkbox_label,
        value=cfg.log_y_default,
        key=f"{wp}_log_y_{chk_ns}",
        help=cfg.log_checkbox_help,
    )
    if wp == "iv":
        st.toggle(
            "Color traces by IV curve similarity",
            value=False,
            key=f"{wp}_color_iv_similarity_{chk_ns}",
            help="Groups traces by cumulative pointwise difference between IV curves on overlap voltage.",
        )
        if st.session_state.get(f"{wp}_color_iv_similarity_{chk_ns}", False):
            st.number_input(
                "IV similarity threshold",
                min_value=0.0,
                value=1e-3,
                step=1e-3,
                format="%.6f",
                key=f"{wp}_iv_area_threshold_{chk_ns}",
                help="Devices connect when the chosen pairwise IV distance metric is <= this threshold.",
            )
    if wp == "ret":
        st.checkbox(
            "Relative retention (%)",
            value=False,
            key=f"{wp}_relative_retention_{chk_ns}",
            help="Normalize each selected retention trace to its first valid value = 100%.",
        )
        st.checkbox(
            "Derive resistance from conductance (R = 1/G)",
            value=False,
            key=f"{wp}_derive_resistance_{chk_ns}",
            help="For conductance-like traces (e.g. G... or *(S)), plot derived resistance.",
        )

    x_col = str(st.session_state[xa_key])
    y_cols = numeric_y_columns(df, x_col)
    if not y_cols:
        st.warning("No plottable numeric series found (excluding the X column).")
        return

    ns = chk_ns
    chk_prefix = f"{wp}_chk_{ns}_"
    use_checkboxes = len(y_cols) <= 20
    grid_map = crossbar_column_map(y_cols)
    other_y = [c for c in y_cols if c not in set(grid_map.values())]
    other_key = f"{wp}_other_series_{ns}"
    sel_key = f"{wp}_series_multiselect_{ns}"

    st.divider()
    st.subheader("Series to plot")
    use_grid_ui = False
    if grid_map:
        use_grid_ui = st.checkbox(
            "Pick devices on 16×16 crossbar",
            value=True,
            key=f"{wp}_use_crossbar_grid_{ns}",
        )

    if grid_map and use_grid_ui:
        st.caption(
            f"Column names like `{cfg.crossbar_example}` or `{cfg.crossbar_example.replace(':', '-')}` → row 3, column 0. "
            "The **main area** shows the grid: **horizontal = row**, **vertical = column**. "
            "Empty cells are not in this file."
        )
        ga, gc = st.columns(2)
        if ga.button("Crossbar: all in file", use_container_width=True, key=f"{wp}_cx_all_{ns}"):
            for r in range(GRID_SIZE):
                for c in range(GRID_SIZE):
                    if (r, c) in grid_map:
                        st.session_state[f"{wp}_cx_{ns}_{r}_{c}"] = True
            st.rerun()
        if gc.button("Crossbar: clear", use_container_width=True, key=f"{wp}_cx_clear_{ns}"):
            for r in range(GRID_SIZE):
                for c in range(GRID_SIZE):
                    if (r, c) in grid_map:
                        st.session_state[f"{wp}_cx_{ns}_{r}_{c}"] = False
            st.rerun()

        if other_y:
            if other_key not in st.session_state:
                st.session_state[other_key] = list(other_y)
            st.multiselect(
                "Traces without crossbar coordinates",
                options=other_y,
                key=other_key,
            )
    elif use_checkboxes:
        c1, c2 = st.columns(2)
        if c1.button("Select all", use_container_width=True, key=f"{wp}_sel_all_{ns}"):
            for c in y_cols:
                st.session_state[f"{chk_prefix}{c}"] = True
            st.rerun()
        if c2.button("Clear", use_container_width=True, key=f"{wp}_sel_clear_{ns}"):
            for c in y_cols:
                st.session_state[f"{chk_prefix}{c}"] = False
            st.rerun()

        for c in y_cols:
            key = f"{chk_prefix}{c}"
            if key not in st.session_state:
                st.session_state[key] = True
            st.checkbox(c, key=key)
    else:
        if sel_key not in st.session_state:
            st.session_state[sel_key] = list(y_cols)

        valid = [c for c in st.session_state[sel_key] if c in y_cols]
        if len(valid) != len(st.session_state[sel_key]) or not valid:
            st.session_state[sel_key] = list(y_cols)

        c1, c2 = st.columns(2)
        if c1.button("Select all", use_container_width=True, key=f"{wp}_ms_sel_all_{ns}"):
            st.session_state[sel_key] = list(y_cols)
            st.rerun()
        if c2.button("Clear", use_container_width=True, key=f"{wp}_ms_sel_clear_{ns}"):
            st.session_state[sel_key] = []
            st.rerun()

        st.multiselect(
            "Choose columns",
            options=y_cols,
            key=sel_key,
        )

    if not (grid_map and use_grid_ui):
        st.caption(
            f"Tip: with many columns (>20), series are chosen via the multiselect. "
            f"With 20 or fewer, each series has its own checkbox. "
            f"Names like `{cfg.crossbar_example}` enable the 16×16 crossbar picker for **{cfg.trace_kind_tip}**."
        )


def _wide_csv_main_plot(cfg: WideCsvViewConfig) -> None:
    """Main area: left = crossbar + data preview; right = chart (sidebar unchanged)."""
    wp = cfg.wp
    if cfg.df_key not in st.session_state:
        st.info(cfg.empty_info)
        return

    df = st.session_state[cfg.df_key]
    all_cols = list(df.columns)
    xa_key = _x_axis_state_key(wp)
    x_col = str(st.session_state.get(xa_key, all_cols[0]))
    if x_col not in all_cols:
        x_col = all_cols[0]

    chk_ns = str(st.session_state[cfg.chk_ns_key])
    log_y_axis = bool(st.session_state.get(f"{wp}_log_y_{chk_ns}", cfg.log_y_default))

    y_cols = numeric_y_columns(df, x_col)
    if not y_cols:
        st.warning("No plottable numeric series found (excluding the X column).")
        return

    ns = chk_ns
    chk_prefix = f"{wp}_chk_{ns}_"
    use_checkboxes = len(y_cols) <= 20
    grid_map = crossbar_column_map(y_cols)
    other_y = [c for c in y_cols if c not in set(grid_map.values())]
    other_key = f"{wp}_other_series_{ns}"
    sel_key = f"{wp}_series_multiselect_{ns}"

    use_grid_ui = bool(st.session_state.get(f"{wp}_use_crossbar_grid_{ns}", True)) if grid_map else False

    selected: list[str] = []
    if grid_map and use_grid_ui:
        for (r, c) in sorted(grid_map):
            cx_key = f"{wp}_cx_{ns}_{r}_{c}"
            if st.session_state.get(cx_key, True):
                selected.append(grid_map[(r, c)])
        if other_y:
            selected.extend(x for x in st.session_state.get(other_key, []) if x in other_y)
    elif use_checkboxes:
        for c in y_cols:
            if st.session_state.get(f"{chk_prefix}{c}", True):
                selected.append(c)
    else:
        selected = [x for x in st.session_state.get(sel_key, []) if x in y_cols]

    chart_pct = st.slider(
        "Chart panel width (%)",
        min_value=35,
        max_value=75,
        value=55,
        key=f"{wp}_chart_col_pct",
        help="Adjust how much horizontal space the plot uses (remainder = crossbar and data preview).",
    )
    left_w = 100 - chart_pct
    left_col, chart_col = st.columns([left_w, chart_pct])

    with left_col:
        if wp == "iv" and y_cols and not grid_map:
            st.info(
                "No **16×16 crossbar** column names were found. Expected patterns like `I3:0(A)`, `G3:0(I)`, or hyphen forms "
                "(row and column 0–15; **I** = per-cell current, **G** = conductance/state). Use the **sidebar** multiselect or per-series checkboxes."
            )

        if grid_map and use_grid_ui:
            render_crossbar_checkbox_grid(grid_map, wp=wp, ns=ns)

        if not selected:
            st.warning("Select at least one series to plot.")

        with st.expander("Data preview"):
            st.dataframe(df.head(50), use_container_width=True)

    with chart_col:
        if not selected:
            return

        only_last = st.toggle(
            "Only show last selected trace",
            value=False,
            key=f"{wp}_only_last_{ns}",
        )
        plot_selected = selected[-1:] if only_last else selected
        derive_resistance = bool(st.session_state.get(f"{wp}_derive_resistance_{ns}", False)) if wp == "ret" else False
        color_close_starts = False
        close_threshold = 10
        color_iv_similarity = False
        iv_area_threshold = 1e-3
        if wp == "ret":
            color_close_starts = st.toggle(
                "Color traces by close start values",
                value=False,
                key=f"{wp}_color_close_starts_{ns}",
            )
            if color_close_starts:
                close_threshold = st.slider(
                    "Close-start threshold (%)",
                    min_value=1,
                    max_value=50,
                    value=10,
                    key=f"{wp}_close_start_threshold_{ns}",
                    help="Traces whose first valid values differ by <= threshold% are colored together.",
                )
        elif wp == "iv":
            color_iv_similarity = bool(st.session_state.get(f"{wp}_color_iv_similarity_{ns}", False))
            iv_area_threshold = float(st.session_state.get(f"{wp}_iv_area_threshold_{ns}", 1e-3))

        relative_retention = bool(st.session_state.get(f"{wp}_relative_retention_{ns}", False)) if wp == "ret" else False
        plot_df = df
        selected_conductance = [col for col in plot_selected if is_conductance_column(col)]
        y_label = "Conductance" if wp == "ret" and selected_conductance else "Resistance"

        if derive_resistance:
            plot_df = df.copy()
            for col in selected_conductance:
                g = pd.to_numeric(df[col], errors="coerce")
                plot_df[col] = (1.0 / g).where(g != 0)
            y_label = "Resistance"

        if relative_retention:
            if plot_df is df:
                plot_df = df.copy()
            for col in plot_selected:
                plot_df[col] = relative_to_first_valid_percent(plot_df[col])
            y_label = "Retention (%)"

        trace_color_map: dict[str, str] | None = None
        groups: dict[str, int] = {}
        group_color_map: dict[int, str] = {}
        show_group_summary = False
        if color_close_starts:
            groups = close_start_groups(df, plot_selected, threshold_percent=float(close_threshold))
            palette = qualitative.Plotly
            trace_color_map = {}
            for name in plot_selected:
                gid = groups.get(name)
                if gid is None:
                    continue
                group_color_map[gid] = palette[gid % len(palette)]
                trace_color_map[name] = group_color_map[gid]
            if groups:
                st.caption(f"Close-start groups: **{len(set(groups.values()))}** at threshold **{close_threshold}%**.")
                st.caption(
                    "Grouping formula: for starts `a` and `b`, "
                    "`percent_diff = |a - b| / max(|a|, |b|, 1e-15) × 100`; "
                    "traces are grouped when adjacent sorted starts are within the threshold."
                )
                ordered_group_ids = sorted(set(groups.values()))
                group_counts = {gid: sum(1 for name in plot_selected if groups.get(name) == gid) for gid in ordered_group_ids}
                group_labels = [f"Group {gid + 1} ({group_counts[gid]} devices)" for gid in ordered_group_ids]
                group_label_to_id = {label: gid for label, gid in zip(group_labels, ordered_group_ids, strict=False)}
                groups_key = f"{wp}_picked_groups_{ns}"
                current = st.session_state.get(groups_key, group_labels)
                valid = [g for g in current if g in group_labels]
                if not valid:
                    valid = group_labels
                st.session_state[groups_key] = valid
                picked_labels = st.multiselect(
                    "Show groups",
                    options=group_labels,
                    key=groups_key,
                    help="Pick which close-start groups to display.",
                )
                picked_ids = {group_label_to_id[g] for g in picked_labels}
                browse_key = f"{wp}_browse_groups_{ns}"
                browse_on = st.toggle(
                    "Browse one group at a time",
                    value=False,
                    key=browse_key,
                    help="Preview a single group without changing the active Show groups selection.",
                )
                if browse_on:
                    browse_options = group_labels
                    browse_pick = st.selectbox(
                        "Browse group",
                        options=browse_options,
                        key=f"{wp}_browse_group_pick_{ns}",
                    )
                    browse_gid = group_label_to_id.get(browse_pick)
                    picked_ids = set() if browse_gid is None else {browse_gid}
                if picked_ids:
                    plot_selected = [name for name in plot_selected if groups.get(name) in picked_ids]
                else:
                    plot_selected = []
                if trace_color_map is not None:
                    trace_color_map = {name: color for name, color in trace_color_map.items() if name in set(plot_selected)}
                show_group_summary = st.toggle(
                    "Show group average + min/max band",
                    value=True,
                    key=f"{wp}_show_group_summary_{ns}",
                    help="Average is a line; min-max is shown as a light band for each displayed group.",
                )
        elif color_iv_similarity:
            rows_for_distance = [{"device": name, "iv_col": name} for name in plot_selected]
            similarity_cache_key = f"{wp}_iv_similarity_cache_{ns}"
            similarity_signature = (
                tuple(plot_selected),
                str(x_col),
                "sum",
            )
            cached = st.session_state.get(similarity_cache_key)
            if isinstance(cached, dict) and cached.get("signature") == similarity_signature:
                devices = list(cached["devices"])
                dist = np.array(cached["dist"], dtype=float)
            else:
                devices, dist = iv_pairwise_distance_matrix(
                    rows_for_distance,
                    plot_df,
                    plot_df[x_col],
                    metric="sum",
                )
                st.session_state[similarity_cache_key] = {
                    "signature": similarity_signature,
                    "devices": list(devices),
                    "dist": np.asarray(dist, dtype=float),
                }
            groups = groups_from_pairwise_distances(devices, dist, threshold=iv_area_threshold)
            palette = qualitative.Plotly
            trace_color_map = {}
            for name in plot_selected:
                gid = groups.get(name)
                if gid is None:
                    continue
                group_color_map[gid] = palette[gid % len(palette)]
                trace_color_map[name] = group_color_map[gid]
            if groups:
                n_groups = len(set(groups.values()))
                st.caption(
                    f"IV similarity groups: **{n_groups}** using **sum** "
                    f"at threshold **{iv_area_threshold:.6g} A**."
                )
                st.caption(
                    "Grouping formula: for each pair, compute distance on interpolated overlap-voltage grid "
                    "(sum: Σ|ΔI|); "
                    "traces are grouped when pairwise distance <= threshold and connected via single-link chaining."
                )
                ordered_group_ids = sorted(set(groups.values()))
                group_counts = {
                    gid: sum(1 for name in plot_selected if groups.get(name) == gid) for gid in ordered_group_ids
                }
                group_labels = [f"Group {gid + 1} ({group_counts[gid]} devices)" for gid in ordered_group_ids]
                group_label_to_id = {label: gid for label, gid in zip(group_labels, ordered_group_ids, strict=False)}
                groups_key = f"{wp}_picked_groups_{ns}"
                current = st.session_state.get(groups_key, group_labels)
                valid = [g for g in current if g in group_labels]
                if not valid:
                    valid = group_labels
                st.session_state[groups_key] = valid
                picked_labels = st.multiselect(
                    "Show groups",
                    options=group_labels,
                    key=groups_key,
                    help="Pick which IV similarity groups to display.",
                )
                picked_ids = {group_label_to_id[g] for g in picked_labels}
                browse_key = f"{wp}_browse_groups_{ns}"
                browse_on = st.toggle(
                    "Browse one group at a time",
                    value=False,
                    key=browse_key,
                    help="Preview a single group without changing the active Show groups selection.",
                )
                if browse_on:
                    browse_options = group_labels
                    browse_pick = st.selectbox(
                        "Browse group",
                        options=browse_options,
                        key=f"{wp}_browse_group_pick_{ns}",
                    )
                    browse_gid = group_label_to_id.get(browse_pick)
                    picked_ids = set() if browse_gid is None else {browse_gid}
                if picked_ids:
                    plot_selected = [name for name in plot_selected if groups.get(name) in picked_ids]
                else:
                    plot_selected = []
                if trace_color_map is not None:
                    trace_color_map = {name: color for name, color in trace_color_map.items() if name in set(plot_selected)}
                show_group_summary = st.toggle(
                    "Show group average + min/max band",
                    value=False,
                    key=f"{wp}_show_group_summary_{ns}",
                    help="Average is a line; min-max is shown as a light band for each displayed group.",
                )

        if not plot_selected:
            st.warning("No traces left after group filtering.")
            return

        export_x, _ = prepare_x_axis(plot_df, x_col)
        export_df = pd.DataFrame({x_col: export_x})
        for col in plot_selected:
            export_df[col] = pd.to_numeric(plot_df[col], errors="coerce")
        st.download_button(
            "Export clean data as CSV",
            data=export_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{wp}_clean_{ns}.csv",
            mime="text/csv",
            help="Exports the currently filtered traces (after selection, group filtering, and retention transforms).",
            key=f"{wp}_export_clean_{ns}",
        )

        if log_y_axis:
            bad = False
            for col in plot_selected:
                s = pd.to_numeric(plot_df[col], errors="coerce")
                if cfg.log_y_use_abs_y:
                    s_plot = s.abs()
                    if s_plot.notna().any() and (s_plot.dropna() == 0).any():
                        bad = True
                        break
                else:
                    if s.notna().any() and (s.dropna() <= 0).any():
                        bad = True
                        break
            if bad:
                if relative_retention:
                    st.warning(
                        "Some selected relative retention values are **≤ 0**; they cannot be shown on a log axis "
                        "and may disappear. Turn off logarithmic Y to see them."
                    )
                elif derive_resistance:
                    st.warning(
                        "Some derived resistance values are **≤ 0** (or undefined from G=0); they cannot be shown on a log axis "
                        "and may disappear. Turn off logarithmic Y to see them."
                    )
                else:
                    st.warning(cfg.non_positive_log_warning)

        lines_to_plot = [] if show_group_summary else plot_selected
        fig = plot_lines(
            plot_df,
            x_col,
            lines_to_plot,
            log_y=log_y_axis,
            y_quantity_label=y_label,
            log_y_use_abs_y=cfg.log_y_use_abs_y,
            trace_color_map=trace_color_map,
        )
        if show_group_summary and groups:
            add_group_summary_overlays(
                fig,
                plot_df,
                x_col,
                plot_selected,
                groups,
                group_color_map,
                log_y=log_y_axis,
                log_y_use_abs_y=cfg.log_y_use_abs_y,
            )
        st.plotly_chart(fig, use_container_width=True)
        if log_y_axis:
            y_flat: list[float] = []
            for c in plot_selected:
                s = pd.to_numeric(plot_df[c], errors="coerce").dropna()
                if cfg.log_y_use_abs_y:
                    s = s.abs()
                y_flat.extend(float(x) for x in s if float(x) > 0)
            if y_flat:
                ymin, ymax = min(y_flat), max(y_flat)
                ratio = ymax / ymin
                abs_note = "Plotted Y is **|current|**. " if cfg.log_y_use_abs_y else ""
                st.caption(
                    f"**Y is logarithmic (base 10).** {abs_note}"
                    f"Selected positive values span about **×{ratio:.2f}** "
                    f"({ymin:.2e} … {ymax:.2e}). If that ratio is small, curves look almost like a linear axis; "
                    f"the Y grid is still **multiplicative** (see power-of-10 ticks)."
                )

        if (color_close_starts or color_iv_similarity) and groups:
            group_to_cols: dict[int, list[str]] = {}
            for col in plot_selected:
                gid = groups.get(col)
                if gid is None:
                    continue
                group_to_cols.setdefault(gid, []).append(col)
            render_group_info(plot_df, x_col, group_to_cols, value_label=y_label, key=f"{wp}_{ns}")


# Dedicated keys so the combined view never collides with the standalone retention/IV modes
# (which expect companion keys like ``_retention_chk_ns`` whenever their df key is present).
CORR_RET_DF_KEY = "corr_retention_df"
CORR_IV_DF_KEY = "corr_iv_df"
CORR_RET_CSV_ID = "_corr_ret_csv_id"
CORR_IV_CSV_ID = "_corr_iv_csv_id"


def _parse_uploaded_csv(uploaded: Any) -> pd.DataFrame | None:
    """Parse an uploaded CSV with ``read_csv_bytes``; surface parse/empty errors in the UI."""
    try:
        df = read_csv_bytes(uploaded.getvalue())
    except Exception as exc:  # noqa: BLE001 — surface parse errors in UI
        st.error(f"Could not parse CSV: {exc}")
        return None
    if df.empty:
        st.warning("The CSV has no rows.")
        return None
    return df


def _correlation_sidebar_controls() -> None:
    """Two uploaders + read settings for the combined retention/IV correlation view."""
    ret_file = st.file_uploader("Upload retention CSV", type=["csv"], key="corr_ret_uploader")
    iv_file = st.file_uploader("Upload IV CSV", type=["csv"], key="corr_iv_uploader")

    if ret_file is not None:
        rid = (ret_file.name, len(ret_file.getvalue()))
        if st.session_state.get(CORR_RET_CSV_ID) != rid:
            df = _parse_uploaded_csv(ret_file)
            if df is not None:
                st.session_state[CORR_RET_DF_KEY] = df
                st.session_state[CORR_RET_CSV_ID] = rid

    if iv_file is not None:
        iid = (iv_file.name, len(iv_file.getvalue()))
        if st.session_state.get(CORR_IV_CSV_ID) != iid:
            df = _parse_uploaded_csv(iv_file)
            if df is not None:
                st.session_state[CORR_IV_DF_KEY] = df
                st.session_state[CORR_IV_CSV_ID] = iid

    have_ret = CORR_RET_DF_KEY in st.session_state
    have_iv = CORR_IV_DF_KEY in st.session_state
    if not (have_ret and have_iv):
        st.info(
            "Upload **both** a retention CSV (columns like `G3:0(S)`) and an IV CSV (columns like `I3:0(A)`). "
            "Devices are matched by crossbar cell `(row, col)`. The SET read uses the **decreasing** branch "
            "of the chosen polarity at the read voltage."
        )
        return

    iv_df = st.session_state[CORR_IV_DF_KEY]
    ret_df = st.session_state[CORR_RET_DF_KEY]
    iv_cols = list(iv_df.columns)
    ret_cols = list(ret_df.columns)

    st.radio(
        "Group devices by",
        ["Retention", "IV read resistance", "IV curve similarity"],
        key="corr_group_by",
        help="The chosen measurement defines the groups; both charts are then colored by those groups.",
    )

    v_guess = infer_voltage_column(iv_cols)
    st.selectbox(
        "IV voltage column",
        options=iv_cols,
        index=iv_cols.index(v_guess) if v_guess in iv_cols else 0,
        key="corr_iv_vcol",
    )
    x_guess = infer_x_column(ret_cols)
    st.selectbox(
        "Retention X axis (time)",
        options=ret_cols,
        index=ret_cols.index(x_guess) if x_guess in ret_cols else 0,
        key="corr_ret_xcol",
    )

    st.radio(
        "SET polarity",
        ["positive", "negative"],
        key="corr_polarity",
        help="Positive reads the 0\u2190+Vmax return branch; negative reads the 0\u2190\u2212Vmax return branch.",
    )
    st.number_input(
        "Read voltage magnitude (V)",
        min_value=0.0,
        value=0.2,
        step=0.05,
        key="corr_read_mag",
        help="Magnitude only; the sign is taken from the chosen SET polarity. Used for IV read-resistance grouping.",
    )
    st.slider(
        "Grouping threshold (%)",
        min_value=1,
        max_value=50,
        value=10,
        key="corr_thr",
        help="Devices whose grouping values differ by <= threshold% are grouped (and colored) together.",
    )
    st.number_input(
        "IV similarity threshold",
        min_value=0.0,
        value=1e-3,
        step=1e-3,
        format="%.6f",
        key="corr_iv_area_thr",
        help="Devices connect when the chosen pairwise IV distance metric is <= this threshold.",
    )
    st.divider()
    st.checkbox("Logarithmic retention Y", value=True, key="corr_ret_log")
    st.checkbox(
        "Derive resistance from conductance (R = 1/G)",
        value=False,
        key="corr_derive_resistance",
        help="For conductance-like retention traces (e.g. G... or *(S)), plot derived resistance on the retention chart.",
    )
    st.checkbox(
        "Logarithmic IV current (Y)",
        value=False,
        key="corr_iv_log",
        help="Log Y plots **|current|** so positive and negative branches are visible.",
    )


_UNGROUPED_COLOR = "#9e9e9e"


def _group_palette_color(gid: int) -> str:
    return qualitative.Plotly[gid % len(qualitative.Plotly)]


def _build_retention_iv_device_rows(
    ret_df: pd.DataFrame,
    iv_df: pd.DataFrame,
    common_cells: list[tuple[int, int]],
    ret_map: dict[tuple[int, int], str],
    iv_map: dict[tuple[int, int], str],
    voltage_series: Any,
    *,
    polarity: str,
    read_voltage: float,
) -> list[dict[str, Any]]:
    """Match crossbar devices and compute retention start, IV read R, and conductance metrics."""
    rows: list[dict[str, Any]] = []
    for (r, c) in common_cells:
        ret_col = ret_map[(r, c)]
        iv_col = iv_map[(r, c)]
        ret_start = first_valid_numeric_value(ret_df[ret_col])
        iv_r = iv_read_resistance(
            voltage_series,
            iv_df[iv_col],
            polarity=polarity,
            read_voltage=read_voltage,
        )
        g_t0, _g_final, retention = _retention_conductance_metrics(ret_df, ret_col)
        g_iv = (1.0 / iv_r) if iv_r is not None and iv_r != 0 else None
        rows.append(
            {
                "device": f"r{r}:c{c}",
                "row": r,
                "col": c,
                "ret_col": ret_col,
                "iv_col": iv_col,
                "retention_start": ret_start,
                "iv_read_R": iv_r,
                "G_t0": g_t0,
                "G_iv": g_iv,
                "retention": retention,
            }
        )
    return rows


def _analysis_metrics_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Devices with finite G_t0, G_iv, and end-point retention (%)."""
    records = [
        {
            "device": row["device"],
            "G_t0": row["G_t0"],
            "G_iv": row["G_iv"],
            "retention": row["retention"],
        }
        for row in rows
        if row.get("G_t0") is not None
        and row.get("G_iv") is not None
        and row.get("retention") is not None
        and np.isfinite(row["G_t0"])
        and np.isfinite(row["G_iv"])
        and np.isfinite(row["retention"])
        and row["G_t0"] > 0
        and row["G_iv"] > 0
    ]
    return pd.DataFrame(records)


def _correlation_results_table(metrics: pd.DataFrame) -> pd.DataFrame:
    """Pearson, Spearman, and partial coefficients for the analysis pairs."""
    g_t0 = metrics["G_t0"].to_numpy(dtype=float)
    g_iv = metrics["G_iv"].to_numpy(dtype=float)
    retention = metrics["retention"].to_numpy(dtype=float)

    def _row(pair: str, x: np.ndarray, y: np.ndarray, *, partial_z: np.ndarray | None = None) -> dict[str, Any]:
        r_p, n = _pearson(x, y)
        r_s, _ = _spearman(x, y)
        partial = _partial_correlation_pearson(x, y, partial_z) if partial_z is not None else None
        return {
            "Pair": pair,
            "n": n,
            "Pearson r": None if r_p is None else round(r_p, 4),
            "Spearman rho": None if r_s is None else round(r_s, 4),
            "Partial r (Pearson)": None if partial is None else round(partial, 4),
        }

    rows_out = [
        _row("G_iv vs retention", g_iv, retention),
        _row("G_t0 vs retention", g_t0, retention),
        _row("G_iv vs G_t0", g_iv, g_t0),
        _row("G_iv vs retention | G_t0", g_iv, retention, partial_z=g_t0),
    ]
    return pd.DataFrame(rows_out)


def _add_scatter_panel(
    fig: go.Figure,
    row: int,
    col: int,
    x: np.ndarray,
    y: np.ndarray,
    *,
    x_title: str,
    y_title: str,
    pearson_r: float | None,
    spearman_r: float | None,
    n: int,
    log_x: bool = False,
    log_y: bool = False,
) -> None:
    x_plot = _maybe_log10(x, use_log=log_x)
    y_plot = _maybe_log10(y, use_log=log_y)
    fig.add_trace(
        go.Scatter(
            x=x_plot,
            y=y_plot,
            mode="markers",
            marker={"size": 8, "opacity": 0.55},
            showlegend=False,
        ),
        row=row,
        col=col,
    )
    if len(x_plot) >= 2:
        coeffs = np.polyfit(x_plot, y_plot, 1)
        x_line = np.linspace(float(np.min(x_plot)), float(np.max(x_plot)), 50)
        y_line = coeffs[0] * x_line + coeffs[1]
        fig.add_trace(
            go.Scatter(
                x=x_line,
                y=y_line,
                mode="lines",
                line={"color": "rgba(220, 50, 50, 0.7)", "width": 2},
                showlegend=False,
            ),
            row=row,
            col=col,
        )
    x_suffix = " (log10)" if log_x else ""
    y_suffix = " (log10)" if log_y else ""
    r_txt = f"r={pearson_r:.3f}" if pearson_r is not None else "r=n/a"
    rho_txt = f"ρ={spearman_r:.3f}" if spearman_r is not None else "ρ=n/a"
    fig.add_annotation(
        text=f"{r_txt}, {rho_txt}, n={n}",
        xref="x domain",
        yref="y domain",
        x=0.02,
        y=0.98,
        xanchor="left",
        yanchor="top",
        showarrow=False,
        font={"size": 11},
        row=row,
        col=col,
    )
    fig.update_xaxes(title_text=x_title + x_suffix, row=row, col=col)
    fig.update_yaxes(title_text=y_title + y_suffix, row=row, col=col)


def _correlation_summary_figure(metrics: pd.DataFrame, *, log_axes: bool) -> go.Figure:
    """2×2 summary: three scatters and a correlation bar chart."""
    g_t0 = metrics["G_t0"].to_numpy(dtype=float)
    g_iv = metrics["G_iv"].to_numpy(dtype=float)
    retention = metrics["retention"].to_numpy(dtype=float)
    n = len(metrics)

    r_iv_ret_p, _ = _pearson(g_iv, retention)
    r_iv_ret_s, _ = _spearman(g_iv, retention)
    r_t0_ret_p, _ = _pearson(g_t0, retention)
    r_t0_ret_s, _ = _spearman(g_t0, retention)
    r_iv_t0_p, _ = _pearson(g_iv, g_t0)
    r_iv_t0_s, _ = _spearman(g_iv, g_t0)
    r_partial = _partial_correlation_pearson(g_iv, retention, g_t0)

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "G_t0 vs G_iv (first point vs IV read)",
            "G_iv vs retention (%)",
            "G_t0 vs retention (%)",
            "Correlation coefficients",
        ),
        horizontal_spacing=0.12,
        vertical_spacing=0.14,
    )

    _add_scatter_panel(
        fig,
        1,
        1,
        g_t0,
        g_iv,
        x_title="G_t0 (S)",
        y_title="G_iv (S)",
        pearson_r=r_iv_t0_p,
        spearman_r=r_iv_t0_s,
        n=n,
        log_x=log_axes,
        log_y=log_axes,
    )
    _add_scatter_panel(
        fig,
        1,
        2,
        g_iv,
        retention,
        x_title="G_iv (S)",
        y_title="Retention (%)",
        pearson_r=r_iv_ret_p,
        spearman_r=r_iv_ret_s,
        n=n,
        log_x=log_axes,
        log_y=False,
    )
    _add_scatter_panel(
        fig,
        2,
        1,
        g_t0,
        retention,
        x_title="G_t0 (S)",
        y_title="Retention (%)",
        pearson_r=r_t0_ret_p,
        spearman_r=r_t0_ret_s,
        n=n,
        log_x=log_axes,
        log_y=False,
    )

    categories = [
        "G_iv~retention",
        "G_t0~retention",
        "G_iv~G_t0",
        "partial G_iv~ret|G_t0",
    ]
    pearson_vals = [r_iv_ret_p, r_t0_ret_p, r_iv_t0_p, r_partial]
    spearman_vals = [r_iv_ret_s, r_t0_ret_s, r_iv_t0_s, None]

    fig.add_trace(
        go.Bar(
            name="Pearson",
            x=categories,
            y=[np.nan if v is None else v for v in pearson_vals],
            marker_color="#636EFA",
        ),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Bar(
            name="Spearman",
            x=categories,
            y=[np.nan if v is None else v for v in spearman_vals],
            marker_color="#EF553B",
        ),
        row=2,
        col=2,
    )
    fig.update_yaxes(title_text="Coefficient", range=[-1.05, 1.05], row=2, col=2)
    fig.update_layout(
        barmode="group",
        height=820,
        title_text=f"Correlation analysis (n={n} devices)",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
    )
    return fig


def _correlation_analysis_tab(rows: list[dict[str, Any]]) -> None:
    """Statistical correlations between G_t0, G_iv, and end-point retention."""
    metrics = _analysis_metrics_dataframe(rows)
    n = len(metrics)
    n_matched = len(rows)
    st.caption(
        f"**{n}** of **{n_matched}** matched devices have finite G_t0, G_iv, and retention "
        f"(retention = 100 × G_final / G_t0 in conductance space)."
    )
    if n < 3:
        st.warning("Need at least **3** devices with valid metrics to compute correlations.")
        return

    results = _correlation_results_table(metrics)
    st.dataframe(results, use_container_width=True, hide_index=True)

    log_axes = st.checkbox(
        "Log axes (G only; retention stays linear)",
        value=False,
        key="corr_analysis_log_g",
        help="Apply log10 to G_t0 and G_iv on scatter plots when all values are positive.",
    )

    fig = _correlation_summary_figure(metrics, log_axes=log_axes)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Per-device metrics"):
        show = metrics.copy()
        for col in ("G_t0", "G_iv"):
            show[col] = show[col].map(lambda v: f"{v:.8g}")
        show["retention"] = show["retention"].map(lambda v: f"{v:.6g}")
        st.dataframe(show, use_container_width=True, hide_index=True)
        st.download_button(
            "Export metrics as CSV",
            data=metrics.to_csv(index=False).encode("utf-8"),
            file_name="correlation_metrics.csv",
            mime="text/csv",
            key="corr_analysis_export",
        )


def _retention_iv_correlation_main() -> None:
    """Match devices, group by the chosen measurement, and show retention + IV charts colored by group."""
    have_ret = CORR_RET_DF_KEY in st.session_state
    have_iv = CORR_IV_DF_KEY in st.session_state
    if not (have_ret and have_iv):
        st.info(
            "Upload **both** a retention CSV and an IV CSV in the sidebar. Devices are matched by crossbar cell, "
            "grouped by the chosen measurement, and shown on retention and IV charts colored by that grouping."
        )
        return

    ret_df = st.session_state[CORR_RET_DF_KEY]
    iv_df = st.session_state[CORR_IV_DF_KEY]

    iv_cols = list(iv_df.columns)
    ret_cols = list(ret_df.columns)
    v_col = str(st.session_state.get("corr_iv_vcol", infer_voltage_column(iv_cols)))
    if v_col not in iv_cols:
        v_col = infer_voltage_column(iv_cols)
    ret_x_col = str(st.session_state.get("corr_ret_xcol", infer_x_column(ret_cols)))
    if ret_x_col not in ret_cols:
        ret_x_col = infer_x_column(ret_cols)

    group_by = str(st.session_state.get("corr_group_by", "Retention"))
    polarity = str(st.session_state.get("corr_polarity", "positive"))
    read_mag = float(st.session_state.get("corr_read_mag", 0.2))
    read_voltage = read_mag if polarity == "positive" else -read_mag
    threshold = float(st.session_state.get("corr_thr", 10))
    iv_area_threshold = float(st.session_state.get("corr_iv_area_thr", 1e-3))
    ret_log = bool(st.session_state.get("corr_ret_log", True))
    iv_log = bool(st.session_state.get("corr_iv_log", False))
    derive_resistance = bool(st.session_state.get("corr_derive_resistance", False))

    ret_map = crossbar_column_map(numeric_y_columns(ret_df, ret_x_col))
    iv_map = crossbar_column_map(numeric_y_columns(iv_df, v_col))

    common_cells = sorted(set(ret_map) & set(iv_map))
    n_ret_only = len(set(ret_map) - set(iv_map))
    n_iv_only = len(set(iv_map) - set(ret_map))

    if not common_cells:
        st.warning(
            "No devices matched between the two files. Retention needs `G<r>:<c>` / `I<r>:<c>` style columns and "
            "IV needs `I<r>:<c>(A)` / `G<r>:<c>` columns mapping to the same `(row, col)` crossbar cells."
        )
        return

    rows = _build_retention_iv_device_rows(
        ret_df,
        iv_df,
        common_cells,
        ret_map,
        iv_map,
        iv_df[v_col],
        polarity=polarity,
        read_voltage=read_voltage,
    )
    table = pd.DataFrame(rows)

    # Group by the chosen measurement; devices missing that value are left ungrouped (gray).
    if group_by == "Retention":
        basis_values = {row["device"]: row["retention_start"] for row in rows if row["retention_start"] is not None}
        basis_label = "retention start value"
        groups = close_value_groups(basis_values, threshold_percent=threshold)
        group_threshold_label = f"{threshold:.0f}%"
    elif group_by == "IV read resistance":
        basis_values = {row["device"]: row["iv_read_R"] for row in rows if row["iv_read_R"] is not None}
        basis_label = f"IV read resistance at {read_voltage:+.3g} V"
        groups = close_value_groups(basis_values, threshold_percent=threshold)
        group_threshold_label = f"{threshold:.0f}%"
    else:
        basis_label = "IV curve similarity (area between curves)"
        corr_similarity_signature = (
            st.session_state.get(CORR_IV_CSV_ID),
            tuple(row["iv_col"] for row in rows),
            str(v_col),
            "sum",
        )
        cached = st.session_state.get("corr_iv_similarity_cache")
        if isinstance(cached, dict) and cached.get("signature") == corr_similarity_signature:
            devices = list(cached["devices"])
            dist = np.array(cached["dist"], dtype=float)
        else:
            devices, dist = iv_pairwise_distance_matrix(rows, iv_df, iv_df[v_col], metric="sum")
            st.session_state["corr_iv_similarity_cache"] = {
                "signature": corr_similarity_signature,
                "devices": list(devices),
                "dist": np.asarray(dist, dtype=float),
            }
        groups = groups_from_pairwise_distances(devices, dist, threshold=iv_area_threshold)
        basis_values = {name: 0.0 for name in devices}
        basis_label = "IV curve similarity (sum)"
        group_threshold_label = f"{iv_area_threshold:.6g} A"

    table["group"] = table["device"].map(lambda d: groups.get(d))
    n_groups = len({g for g in groups.values()})
    n_ungrouped = int(table["group"].isna().sum())

    def _color_for(device: str) -> str:
        gid = groups.get(device)
        return _UNGROUPED_COLOR if gid is None else _group_palette_color(gid)

    st.caption(
        f"Matched **{len(common_cells)}** devices in both files "
        f"(retention-only: {n_ret_only}; IV-only: {n_iv_only})."
    )

    tab_grouped, tab_analysis = st.tabs(["Grouped charts", "Correlation analysis"])

    group_color_map = {gid: _group_palette_color(gid) for gid in sorted(set(groups.values()))}
    ret_conductance_cols = [row["ret_col"] for row in rows if is_conductance_column(row["ret_col"])]
    ret_y_label = "Conductance" if ret_conductance_cols else "Resistance"

    # Retention chart can show derived resistance (R = 1/G) for conductance columns, like retention-only.
    # Grouping still uses the raw retention start value (matching retention mode behavior).
    ret_plot_df = ret_df
    if derive_resistance and ret_conductance_cols:
        ret_plot_df = ret_df.copy()
        for col in ret_conductance_cols:
            g = pd.to_numeric(ret_df[col], errors="coerce")
            ret_plot_df[col] = (1.0 / g).where(g != 0)
        ret_y_label = "Resistance"

    # Group display + summary controls (mirrors the retention close-start group functionality).
    ordered_gids = sorted(set(groups.values()))
    group_counts = {gid: sum(1 for g in groups.values() if g == gid) for gid in ordered_gids}
    group_labels = [f"Group {gid + 1} ({group_counts[gid]} devices)" for gid in ordered_gids]
    label_to_gid = {label: gid for label, gid in zip(group_labels, ordered_gids, strict=False)}
    ungrouped_label = f"Ungrouped ({n_ungrouped} devices)"
    options = group_labels + ([ungrouped_label] if n_ungrouped else [])

    with tab_analysis:
        _correlation_analysis_tab(rows)

    with tab_grouped:
        st.caption(
            f"Grouped **{len(basis_values)}** devices by **{basis_label}** "
            f"into **{n_groups}** group(s) at threshold **{group_threshold_label}** "
            f"(ungrouped/gray: {n_ungrouped})."
        )

        groups_key = "corr_show_groups"
        current = st.session_state.get(groups_key, options)
        valid = [g for g in current if g in options]
        if not valid:
            valid = options
        st.session_state[groups_key] = valid
        picked = st.multiselect(
            "Show groups",
            options=options,
            key=groups_key,
            help="Pick which groups (colors) to display on both charts.",
        )
        picked_gids = {label_to_gid[label] for label in picked if label in label_to_gid}
        show_ungrouped = ungrouped_label in picked
        browse_on = st.toggle(
            "Browse one group at a time",
            value=False,
            key="corr_browse_groups",
            help="Preview one group without changing the active Show groups selection.",
        )
        if browse_on:
            browse_pick = st.selectbox(
                "Browse group",
                options=options,
                key="corr_browse_group_pick",
            )
            if browse_pick == ungrouped_label:
                picked_gids = set()
                show_ungrouped = True
            else:
                gid = label_to_gid.get(browse_pick)
                picked_gids = set() if gid is None else {gid}
                show_ungrouped = False
        show_group_summary = st.toggle(
            "Show group average + min/max band",
            value=(group_by == "IV curve similarity"),
            key="corr_show_summary",
            help="Per group, draw one average line and a light min-max band on both charts (ungrouped traces stay as lines).",
        )

        displayed_rows = [
            row
            for row in rows
            if (groups.get(row["device"]) in picked_gids)
            or (groups.get(row["device"]) is None and show_ungrouped)
        ]
        grouped_rows = [row for row in displayed_rows if groups.get(row["device"]) is not None]
        ungrouped_rows = [row for row in displayed_rows if groups.get(row["device"]) is None]

        if not displayed_rows:
            st.warning("No groups selected to display.")
            return

        def _render_chart(
            *,
            df: pd.DataFrame,
            x_col: str,
            col_key: str,
            log_y: bool,
            y_label: str,
            use_abs_y: bool,
        ) -> go.Figure:
            if show_group_summary:
                base_cols = [row[col_key] for row in ungrouped_rows]
                fig = plot_lines(
                    df,
                    x_col,
                    base_cols,
                    log_y=log_y,
                    y_quantity_label=y_label,
                    log_y_use_abs_y=use_abs_y,
                    trace_color_map={row[col_key]: _UNGROUPED_COLOR for row in ungrouped_rows},
                )
                grouped_cols = [row[col_key] for row in grouped_rows]
                col_groups = {row[col_key]: groups[row["device"]] for row in grouped_rows}
                add_group_summary_overlays(
                    fig,
                    df,
                    x_col,
                    grouped_cols,
                    col_groups,
                    group_color_map,
                    log_y=log_y,
                    log_y_use_abs_y=use_abs_y,
                )
                return fig
            cols = [row[col_key] for row in displayed_rows]
            return plot_lines(
                df,
                x_col,
                cols,
                log_y=log_y,
                y_quantity_label=y_label,
                log_y_use_abs_y=use_abs_y,
                trace_color_map={row[col_key]: _color_for(row["device"]) for row in displayed_rows},
            )

        left_col, right_col = st.columns(2)
        with left_col:
            st.subheader("Retention")
            st.plotly_chart(
                _render_chart(
                    df=ret_plot_df,
                    x_col=ret_x_col,
                    col_key="ret_col",
                    log_y=ret_log,
                    y_label=ret_y_label,
                    use_abs_y=False,
                ),
                use_container_width=True,
            )
        with right_col:
            st.subheader("IV characteristics")
            st.plotly_chart(
                _render_chart(
                    df=iv_df,
                    x_col=v_col,
                    col_key="iv_col",
                    log_y=iv_log,
                    y_label="Current",
                    use_abs_y=True,
                ),
                use_container_width=True,
            )

        st.caption(
            f"Both charts are colored by the **{group_by}** grouping (same color = same group). "
            "Gray traces are devices without a usable grouping value."
        )

        ret_group_to_cols: dict[int, list[str]] = {}
        for row in grouped_rows:
            ret_group_to_cols.setdefault(groups[row["device"]], []).append(row["ret_col"])
        read_resistances = {row["ret_col"]: row["iv_read_R"] for row in grouped_rows if row["iv_read_R"] is not None}
        render_group_info(
            ret_plot_df,
            ret_x_col,
            ret_group_to_cols,
            value_label=ret_y_label,
            key="corr",
            read_resistances=read_resistances,
        )

        with st.expander("Per-device groups and values"):
            show = table[["device", "row", "col", "group", "retention_start", "iv_read_R"]].copy()
            show["group"] = show["group"].map(lambda g: "ungrouped" if pd.isna(g) else int(g) + 1)
            show["grouping_basis"] = group_by
            show["grouping_threshold"] = group_threshold_label
            st.dataframe(show, use_container_width=True)
            st.download_button(
                "Export grouping table as CSV",
                data=show.to_csv(index=False).encode("utf-8"),
                file_name="retention_iv_groups.csv",
                mime="text/csv",
                key="corr_export",
            )


def main() -> None:
    st.set_page_config(page_title="Resistance retention viewer", layout="wide", initial_sidebar_state="expanded")
    st.title("Resistance retention viewer")

    with st.sidebar:
        mode = st.radio(
            "Measurement",
            ["Resistance retention", "IV characteristics", "Retention \u2194 IV correlation"],
            label_visibility="visible",
        )
        st.divider()
        if mode == "Retention \u2194 IV correlation":
            _correlation_sidebar_controls()
        else:
            cfg = RETENTION_VIEW if mode == "Resistance retention" else IV_VIEW
            _wide_csv_sidebar_controls(cfg)

    if mode == "Retention \u2194 IV correlation":
        _retention_iv_correlation_main()
    else:
        cfg = RETENTION_VIEW if mode == "Resistance retention" else IV_VIEW
        _wide_csv_main_plot(cfg)


if __name__ == "__main__":
    main()
