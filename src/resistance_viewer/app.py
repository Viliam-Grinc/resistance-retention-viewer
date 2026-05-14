"""Streamlit UI: wide CSV → resistance vs time and IV (current vs voltage) with per-series visibility."""

from __future__ import annotations

import io
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pandas as pd
import plotly.graph_objects as go
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


def read_csv_bytes(raw: bytes) -> pd.DataFrame:
    """
    Load instrument-style CSV: comma or semicolon separated, scientific notation (e.g. 1.76E-06).

    Picks the parse with the most columns so a wrong delimiter (one fat column) loses.
    """
    text = _decode_csv_text(raw)
    read_kw: list[dict[str, Any]] = [
        {"sep": None, "engine": "python"},
        {"sep": ";", "engine": "python"},
        {"sep": ",", "engine": "python"},
    ]
    best: pd.DataFrame | None = None
    best_n = 0
    last_exc: Exception | None = None
    for kw in read_kw:
        try:
            df = pd.read_csv(io.StringIO(text), **kw)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            continue
        df = cleanup_df(df)
        n = len(df.columns)
        if n > best_n:
            best = df
            best_n = n
    if best is None or best_n < 1:
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
) -> go.Figure:
    x_series, x_title = prepare_x_axis(df, x_col)
    fig = go.Figure()
    for name in y_cols:
        y_raw = df[name]
        y_series = y_raw if pd.api.types.is_numeric_dtype(y_raw) else pd.to_numeric(y_raw, errors="coerce")
        fig.add_trace(
            go.Scatter(
                x=x_series,
                y=y_series,
                mode="lines",
                name=name,
                connectgaps=False,
            )
        )
    yaxis_title = f"{y_quantity_label} (log scale)" if log_y else y_quantity_label
    fig.update_layout(
        margin=dict(l=40, r=24, t=48, b=40),
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
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
    log_y_checkbox_label="Logarithmic resistance (Y)",
    log_y_default=True,
    y_quantity_label="Resistance",
    non_positive_log_warning=(
        "Some selected resistance values are **≤ 0**; they cannot be shown on a log axis "
        "and may disappear. Turn off **Logarithmic resistance (Y)** in the sidebar to see them."
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
        "Some selected current values are **≤ 0**; they cannot be shown on a log axis "
        "and may disappear. Turn off **Logarithmic current (Y)** in the sidebar to see them."
    ),
    log_checkbox_help="Useful when currents span many orders of magnitude.",
    crossbar_example="I3:0(A)",
    trace_kind_tip="current",
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
    """Full-width main area: optional 16×16 crossbar grid, then chart (same flow as retention README)."""
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
    if wp == "iv" and y_cols and not grid_map:
        st.info(
            "No **16×16 crossbar** column names were found. Expected patterns like `I3:0(A)`, `G3:0(I)`, or hyphen forms "
            "(row and column 0–15; **I** = per-cell current, **G** = conductance/state). Use the **sidebar** multiselect or per-series checkboxes."
        )
    other_y = [c for c in y_cols if c not in set(grid_map.values())]
    other_key = f"{wp}_other_series_{ns}"
    sel_key = f"{wp}_series_multiselect_{ns}"

    use_grid_ui = bool(st.session_state.get(f"{wp}_use_crossbar_grid_{ns}", True)) if grid_map else False

    if grid_map and use_grid_ui:
        render_crossbar_checkbox_grid(grid_map, wp=wp, ns=ns)

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

    if not selected:
        st.warning("Select at least one series to plot.")
        return

    if log_y_axis:
        bad = False
        for col in selected:
            s = pd.to_numeric(df[col], errors="coerce")
            if s.notna().any() and (s.dropna() <= 0).any():
                bad = True
                break
        if bad:
            st.warning(cfg.non_positive_log_warning)

    fig = plot_lines(df, x_col, selected, log_y=log_y_axis, y_quantity_label=cfg.y_quantity_label)
    st.plotly_chart(fig, use_container_width=True)
    if log_y_axis:
        y_flat: list[float] = []
        for c in selected:
            s = pd.to_numeric(df[c], errors="coerce").dropna()
            y_flat.extend(float(x) for x in s if float(x) > 0)
        if y_flat:
            ymin, ymax = min(y_flat), max(y_flat)
            ratio = ymax / ymin
            st.caption(
                f"**Y is logarithmic (base 10).** Selected positive values span about **×{ratio:.2f}** "
                f"({ymin:.2e} … {ymax:.2e}). If that ratio is small, curves look almost like a linear axis; "
                f"the Y grid is still **multiplicative** (see power-of-10 ticks)."
            )

    with st.expander("Data preview"):
        st.dataframe(df.head(50), use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="Resistance retention viewer", layout="wide", initial_sidebar_state="expanded")
    st.title("Resistance retention viewer")

    with st.sidebar:
        mode = st.radio(
            "Measurement",
            ["Resistance retention", "IV characteristics"],
            label_visibility="visible",
        )
        st.divider()
        cfg = RETENTION_VIEW if mode == "Resistance retention" else IV_VIEW
        _wide_csv_sidebar_controls(cfg)

    cfg = RETENTION_VIEW if mode == "Resistance retention" else IV_VIEW
    _wide_csv_main_plot(cfg)


if __name__ == "__main__":
    main()
