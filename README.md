# Resistance retention viewer

Small **Streamlit** app that plots resistance traces over time from a **wide** CSV: one column for the X axis (time or index) and **one column per resistance series**. You can plot everything at once or turn individual series on or off.

## Requirements

- Python 3.10+

On macOS, the interpreter is usually **`python3`** (there may be no `python` command). Use **`python3 -m pip`** so you do not rely on a global `pip` on your PATH.

## Setup

From the **project root** (the directory that contains `pyproject.toml` and `requirements.txt`):

```bash
# Only if you are not already there, e.g. after cloning:
# cd path/to/resistance-retention-viewer

python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
python3 -m pip install -r requirements.txt
```

If your shell says `cd: no such file or directory: resistance-retention-viewer`, you are probably **already inside** that folder, or the project lives under a different path—`pwd` and `ls` should show `requirements.txt` in the current directory before you continue.

## Run

### macOS — double-click only (no Terminal window)

**Double-click `Resistance Viewer.app`** in the project folder. That is the only step for normal use.

- In **Finder** you see a single app icon — not a `Contents` folder. That is correct: a `.app` is one **bundle** (a special folder macOS shows as one application). The launcher scripts live inside it; right-click the app → **Show Package Contents** if you ever need to inspect `Contents/MacOS` or `Contents/Resources`.
- After **clone or pull**, run once from Terminal: `./scripts/register_app.sh` (copies the latest `src/` into the `.app` bundle — required because macOS sandbox blocks the app from reading arbitrary folders on double-click).
- You can keep **`Resistance Viewer.app` anywhere** once registered; the runnable code lives inside the bundle under `Contents/Resources/app/`.
- No Terminal window opens; Streamlit runs in the background (you may see a **Python** icon in the Dock while it is running).
- Your browser opens automatically at `http://127.0.0.1:8501` when the server is ready.
- **First run** creates a virtual environment under `~/Library/Application Support/Resistance Viewer/venvs/` (not in the project folder), installs dependencies, and can take a few minutes (macOS notifications; details in `~/Library/Logs/Resistance Viewer/viewer.log`).
- If the viewer is already running, double-clicking again only reopens the browser.
- **Quit**: stop the app from the Dock (right-click the Python icon → Quit), or end the Streamlit process in Activity Monitor.

**If double-click does nothing** (common on first use):

1. Right-click **`Resistance Viewer.app`** → **Open** → confirm **Open** (macOS may block unsigned local apps silently).
2. Or run once: `./scripts/register_app.sh` (clears quarantine and ad-hoc signs the app).

**Troubleshooting:** Check `~/Library/Logs/Resistance Viewer/app-launch.log` (did the app start?) and `viewer.log` (Streamlit). Stop any old server on port 8501 (`lsof -i :8501`) if a previous session is still running.

If you see **“Still starting…”** or a browser timeout notification, that is usually harmless: Streamlit can take a minute on a cold start while Python loads. The app keeps trying to open the browser for several minutes; you can also open `http://127.0.0.1:8501` yourself once the Dock shows Python running.

### Command line

With the virtual environment activated:

```bash
python3 -m streamlit run src/resistance_viewer/app.py
```

If `streamlit` works after install, `streamlit run src/resistance_viewer/app.py` is equivalent.

Open the URL shown in the terminal (usually `http://localhost:8501`).

You can also run the same launcher script without the `.app` bundle:

```bash
./scripts/launch_viewer.sh
```

## CSV format

- **Wide table**: one X column (usually time) and **one column per device / trace**.
- **Delimiter**: **Comma or semicolon** (`;`). Instrument exports (e.g. crossbar retention tests) often use **semicolon-separated** fields.
- **Numbers**: Plain decimals or **scientific notation** (e.g. `1.7560827E-06`). **European decimal comma** is supported when fields are separated by `;` (e.g. `0,1` and `1,5789465E-07`).
- **X axis**: The app prefers columns named like `Time(s)`, `time`, `timestamp`, `date`, or `t` (case-insensitive, including names that start with `time(`); otherwise it defaults to the **first** column.
- **Time values**: Columns that are mostly **numbers** are plotted on a **numeric** X axis (e.g. elapsed **seconds** `0, 900, 1800`). Large values (`>1e9` seconds or `>1e12` ms) are treated as **Unix timestamps** and shown as datetimes. Text columns that parse as dates use a datetime axis.
- **Trailing delimiter**: A stray `;` at the end of a line (empty last column) is dropped when it contains no data.
- **Missing data**: Gaps are shown as breaks in the line (`connectgaps=False`).

### Example (comma-separated)

```csv
time_min,R_sample_A,R_sample_B,R_sample_C
0,100.2,99.8,101.0
10,99.1,99.0,100.4
20,98.5,98.2,99.9
```

### Example (semicolon-separated, scientific notation)

```csv
Time(s);G3:0(S);G4:0(S)
0;1.7560827E-06;1.521082E-06
900;1.5611304E-06;1.4530755E-06
```

## UI behavior

- **Layout:** **Sidebar** = upload and all setup (unchanged). **Main area** = left: 16×16 crossbar (when enabled) and **Data preview**; right: **chart only**. Use **Chart panel width (%)** above the main area to resize the plot column.
- Upload a `.csv` file.
- Choose the **X axis** column in the sidebar.
- **Logarithmic Y** is on by default in retention mode. The axis uses **base-10 log** (tick spacing is multiplicative). If all traces sit in a **narrow factor range** (e.g. only ×1.2 from min to max), the lines can **look almost like a linear plot**—check the power-of-10 style Y labels. Turn log off for a strictly linear axis, or if you have **zero or negative** values in the selection.
- **Conductance vs. derived resistance (retention)**: retention traces named like `G...` / `(S)` are treated as conductance by default. Enable **Derive resistance from conductance (R = 1/G)** in the sidebar to plot derived resistance instead.
- **Retention close-start coloring** (retention mode only): in the chart panel, enable **Color traces by close start values** to color traces with similar first valid values together. Use **Close-start threshold (%)** to define how close starts must be (higher threshold = fewer, larger groups), and **Show groups** to display only selected groups.
- The chart panel displays the **grouping formula** used for close-start grouping so the threshold logic is explicit.
- **Group summary overlay**: with close-start coloring enabled, turn on **Show group average + min/max band** to draw one average line per shown group and a light min–max band around it.
- **Export clean data**: use **Export clean data as CSV** in the chart panel to download currently filtered traces (after selection/group filtering and retention transforms).
- **16×16 crossbar**: If trace columns match `G<row>:<col>(…)` with **row** and **col** in `0…15` (e.g. `G3:0(S)` → row 3, column 0), the sidebar enables **Pick devices on 16×16 crossbar**. The **left main column** shows a **16×16 grid** of checkboxes: **horizontal axis = row**, **vertical axis = column** (matching how many crossbar diagrams are drawn). A dot means that cell is not present in the file. Use **Crossbar: all in file** / **Crossbar: clear** in the sidebar. Traces that do not match the pattern appear under **Traces without crossbar coordinates** in the sidebar.
- If you turn off the crossbar picker (or names do not match), series are chosen the classic way:
  - **20 or fewer** numeric series: **one checkbox per series**, plus **Select all** / **Clear**.
  - **More than 20**: **multiselect** with the same buttons.

Plotly also lets you click legend entries to hide or show traces without changing the sidebar selection.

## Retention ↔ IV correlation

The **Measurement** radio in the sidebar has a third option, **Retention ↔ IV correlation**, that checks whether the groups you see in retention data line up with groups derived from IV (current–voltage) sweeps.

- **Two uploads**: load **both** a retention CSV (columns like `G3:0(S)`) and an IV CSV (columns like `I3:0(A)`). Devices are matched by **crossbar cell** `(row, col)`, so `G3:0(S)` in retention and `I3:0(A)` in IV are treated as the same device. Only devices present in **both** files are analyzed (the app reports how many matched and how many were dropped).
- **IV read voltage and polarity**: pick the **IV voltage column**, the **SET polarity** (positive or negative), and a **read voltage magnitude**. The signed read voltage is `+magnitude` for positive polarity and `−magnitude` for negative.
- **Read branch rule**: an IV sweep has a rising part (`0 → ±Vmax`) and a decreasing/return part (`±Vmax → 0`). The read is taken **only on the decreasing branch** within the chosen polarity (the rising part is ignored), and the current at the read voltage is **linearly interpolated** along that branch (nearest point if the read voltage falls outside the branch range). The per-device read resistance is `R = read_voltage / I(read_voltage)`.
- **Grouping**: retention devices are grouped by their **first valid value** (the existing close-start logic), and IV devices are grouped by **read resistance**. Each side has its own closeness threshold slider (devices within the threshold percent are grouped together).
- **Three correlation views**:
  - **Scatter** of retention start value vs IV read resistance per device, with **Pearson** (linear) and **Spearman** (rank) correlation coefficients. Axes can be log-scaled.
  - **Group-label agreement**: a cross-tab of retention groups vs IV groups plus the **Adjusted Rand Index** (1.0 = identical grouping, ~0 = chance-level, negative = worse than chance).
  - **Crossbar overlap**: two side-by-side 16×16 grids colored by group id (retention vs IV) using the same **X = row**, **Y = column** layout as the device picker, so spatial overlap is easy to eyeball.
- Use the **Per-device table** expander to inspect or export (CSV) the matched values and group assignments.

## Optional install as a package

```bash
python3 -m pip install -e .
```

This does not add a console script; keep using `streamlit run` as above.
