#!/usr/bin/env python
"""Generate reports/status_latest.html — one-page status report for retinal_sim.

Runs pytest, parses PROGRESS.md and CODEREVIEW.md, generates validation
figures, and writes everything to a self-contained HTML file.

Usage:
    python scripts/status_report.py
"""
from __future__ import annotations

import base64
import io
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

# Ensure the repo root is importable without requiring pip install
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# 1. Run pytest
# ---------------------------------------------------------------------------

def run_pytest() -> tuple[str, str, int]:
    """Return (summary_line, failure_details, returncode)."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--tb=short", "-q", "--no-header"],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    output = (result.stdout + result.stderr).strip()
    lines = output.splitlines()

    # Last non-blank line is the summary (e.g. "287 passed, 1 skipped in 25s")
    summary = next((l for l in reversed(lines) if l.strip()), "no output")

    # Collect failure blocks
    failures: list[str] = []
    in_fail = False
    for line in lines:
        if re.match(r"^(FAILED |={3,})", line):
            in_fail = True
        if in_fail:
            failures.append(line)

    return summary, "\n".join(failures), result.returncode


def parse_pytest_summary(summary: str) -> dict:
    """Extract counts from a pytest summary line into a dict."""
    counts: dict[str, int] = {"passed": 0, "failed": 0, "skipped": 0, "error": 0}
    for key in counts:
        m = re.search(rf"(\d+) {key}", summary)
        if m:
            counts[key] = int(m.group(1))
    m = re.search(r"in ([\d.]+)s", summary)
    counts["duration"] = float(m.group(1)) if m else 0.0  # type: ignore[assignment]
    return counts


# ---------------------------------------------------------------------------
# 2. Parse PROGRESS.md
# ---------------------------------------------------------------------------

def parse_phase_table(text: str) -> list[list[str]]:
    """Extract rows from the phase status table."""
    rows: list[list[str]] = []
    in_table = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("| Phase |"):
            in_table = True
            continue
        if not in_table:
            continue
        if stripped.startswith("|---"):
            continue
        if stripped.startswith("|"):
            cells = [c.strip() for c in stripped.split("|")[1:-1]]
            # Keep only numeric Phase rows
            if cells and re.match(r"^\d+$", cells[0]):
                # Strip markdown bold markers
                cells = [re.sub(r"\*\*(.+?)\*\*", r"\1", c) for c in cells]
                rows.append(cells)
        else:
            in_table = False
    return rows


# ---------------------------------------------------------------------------
# 3. Parse CODEREVIEW.md open items
# ---------------------------------------------------------------------------

def parse_open_items(text: str) -> str:
    """Return the raw content of the '## Open Items' section (stripped)."""
    m = re.search(r"## Open Items\n(.*?)(?=\n## |\Z)", text, re.DOTALL)
    if not m:
        return ""
    content = m.group(1).strip()
    if content in ("*(none)*",):
        return ""
    return content


# ---------------------------------------------------------------------------
# 4. Validation figures
# ---------------------------------------------------------------------------

def _fig_to_data_uri(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()


def make_nomogram_figure():
    """Phase 1 validation: Govardovskii nomogram curves."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from retinal_sim.retina.opsin import build_sensitivity_curves, LAMBDA_MAX

    wl = np.arange(380, 721, 1, dtype=float)
    colors = {
        "S_cone": "#5b5bd6",
        "M_cone": "#3aac3a",
        "L_cone": "#d94c4c",
        "rod":    "#888888",
    }
    labels = {"S_cone": "S-cone", "M_cone": "M-cone", "L_cone": "L-cone", "rod": "Rod"}

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for ax, species in zip(axes, ["human", "dog", "cat"]):
        curves = build_sensitivity_curves(species, wl)
        for rtype in ["rod", "S_cone", "M_cone", "L_cone"]:
            if rtype not in curves:
                continue
            lm = LAMBDA_MAX[species][rtype]
            ax.plot(wl, curves[rtype], color=colors[rtype], lw=1.5,
                    label=f"{labels[rtype]} ({lm:.0f} nm)")
        ax.set_title(species.capitalize(), fontsize=10)
        ax.set_xlim(380, 720)
        ax.set_ylim(-0.02, 1.08)
        ax.set_xlabel("Wavelength (nm)", fontsize=9)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, lw=0.3, alpha=0.4)
    axes[0].set_ylabel("Relative sensitivity", fontsize=9)
    fig.suptitle("Phase 1 — Govardovskii Nomogram (A1)", fontsize=11, y=1.01)
    plt.tight_layout()
    return fig


def make_voronoi_figure():
    """Phase 8 validation: Voronoi activation map (synthetic central stimulus)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from retinal_sim.retina.mosaic import PhotoreceptorMosaic
    from retinal_sim.retina.stage import MosaicActivation
    from retinal_sim.output.voronoi import render_voronoi

    rng = np.random.default_rng(42)
    n = 400
    # Denser sampling near centre to mimic foveal concentration
    angles = rng.uniform(0, 2 * np.pi, n)
    radii = rng.beta(1.5, 3.0, n) * 0.45          # beta skewed toward small radii
    pos = np.stack([radii * np.cos(angles),
                    radii * np.sin(angles)], axis=1).astype(np.float32)

    # Human-like 1:16:32 ratio (L:M:S) + rods at periphery
    pool = (["L_cone"] * 32 + ["M_cone"] * 16 + ["S_cone"] * 2 + ["rod"] * 4)
    types_arr = np.array([pool[rng.integers(0, len(pool))] for _ in range(n)], dtype="U10")
    # Peripheral rods
    types_arr[radii > 0.35] = "rod"

    # Response = Gaussian centred stimulus
    responses = np.exp(-np.linalg.norm(pos, axis=1) ** 2 / (2 * 0.12 ** 2)).astype(np.float32)

    mosaic = PhotoreceptorMosaic(
        positions=pos,
        types=types_arr,
        apertures=np.full(n, 5.0, dtype=np.float32),
        sensitivities=np.ones((n, 69), dtype=np.float32),
    )
    act = MosaicActivation(mosaic=mosaic, responses=responses)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))

    # Panel 0: synthetic Gaussian input rendered on a regular grid
    grid_res = 360
    xs = np.linspace(-0.45, 0.45, grid_res)
    ys = np.linspace(-0.45, 0.45, grid_res)
    gx, gy = np.meshgrid(xs, ys)
    gauss_grid = np.exp(-(gx ** 2 + gy ** 2) / (2 * 0.12 ** 2))
    im0 = axes[0].imshow(gauss_grid, origin="lower", cmap="gray", vmin=0, vmax=1,
                         interpolation="bilinear",
                         extent=[-0.45, 0.45, -0.45, 0.45])
    axes[0].set_title("Synthetic input\n(Gaussian, σ = 0.12 a.u.)", fontsize=9)
    axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Panel 1: Voronoi activation
    img = render_voronoi(act, output_size=(360, 360))
    axes[1].imshow(img, origin="lower", interpolation="nearest")
    axes[1].set_title("Voronoi activation\n(colour = type, brightness = response)", fontsize=9)
    axes[1].axis("off")

    # Panel 2: Reconstructed grid
    from retinal_sim.output.reconstruction import render_reconstructed
    recon = render_reconstructed(act, (360, 360))
    im2 = axes[2].imshow(recon, origin="lower", cmap="gray", vmin=0, vmax=1,
                         interpolation="nearest")
    axes[2].set_title("Reconstructed luminance grid\n(nearest-neighbour inverse mapping)", fontsize=9)
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    fig.suptitle("Phase 8 — Voronoi Visualisation (synthetic Gaussian stimulus)", fontsize=11)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 5. HTML assembly
# ---------------------------------------------------------------------------

_STYLE = """
body {
    font-family: system-ui, -apple-system, sans-serif;
    max-width: 980px;
    margin: 40px auto;
    padding: 0 24px;
    color: #222;
    line-height: 1.5;
}
h1 { border-bottom: 2px solid #333; padding-bottom: 6px; }
h2 { border-bottom: 1px solid #ccc; margin-top: 2.2em; color: #333; }
.timestamp { color: #666; font-size: 0.88em; margin-top: -6px; }
.summary-box {
    display: inline-block;
    padding: 10px 18px;
    border-radius: 6px;
    font-size: 1.1em;
    font-weight: bold;
    margin-bottom: 8px;
}
.all-pass { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
.has-fail  { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
table { border-collapse: collapse; width: 100%; margin-top: 8px; }
th, td { border: 1px solid #ddd; padding: 6px 10px; text-align: left; font-size: 0.92em; }
th { background: #f4f4f4; }
tr:nth-child(even) td { background: #fafafa; }
.status-complete { color: #155724; font-weight: bold; }
.status-stub     { color: #856404; }
.status-ns       { color: #999; }
pre {
    background: #f6f6f6;
    border: 1px solid #ddd;
    padding: 12px 14px;
    border-radius: 4px;
    overflow-x: auto;
    font-size: 0.82em;
    white-space: pre-wrap;
    word-break: break-all;
}
.figures { display: flex; flex-wrap: wrap; gap: 18px; margin-top: 10px; }
.figures img { flex: 1 1 420px; max-width: 100%; border: 1px solid #e0e0e0;
               border-radius: 4px; }
.no-open { color: #155724; font-style: italic; }
.open-items { background: #fff3cd; border: 1px solid #ffc107;
              border-radius: 4px; padding: 10px 14px; }
"""


def _status_cell(status: str) -> str:
    lower = status.lower()
    if "complete" in lower:
        return f'<span class="status-complete">{status}</span>'
    if "stub" in lower or "not started" in lower:
        cls = "status-stub" if "stub" in lower else "status-ns"
        return f'<span class="{cls}">{status}</span>'
    return status


def build_html(
    phase_rows: list[list[str]],
    pytest_summary: str,
    pytest_counts: dict,
    pytest_failures: str,
    open_items: str,
    figure_uris: list[tuple[str, str]],   # (caption, data_uri)
) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    ok = pytest_counts["failed"] == 0 and pytest_counts["error"] == 0

    # Test result box
    box_cls = "all-pass" if ok else "has-fail"
    box_text = pytest_summary

    # Phase table
    phase_table_rows = ""
    for cells in phase_rows:
        # cells: [phase, component, status, tests, notes]
        tds = "".join(
            f"<td>{_status_cell(c) if i == 2 else c}</td>"
            for i, c in enumerate(cells)
        )
        phase_table_rows += f"<tr>{tds}</tr>\n"

    # Failures block
    failures_html = ""
    if pytest_failures:
        failures_html = f"<pre>{_escape(pytest_failures)}</pre>"

    # Figures
    figures_html = ""
    if figure_uris:
        imgs = "".join(
            f'<figure style="margin:0; flex:1 1 420px; max-width:100%">'
            f'<img src="{uri}" alt="{cap}">'
            f'<figcaption style="font-size:0.82em;color:#555;margin-top:4px">{cap}</figcaption>'
            f'</figure>'
            for cap, uri in figure_uris
        )
        figures_html = f'<div class="figures">{imgs}</div>'
    else:
        figures_html = "<p><em>No figures generated.</em></p>"

    # Open items
    if open_items:
        items_html = f'<div class="open-items"><pre>{_escape(open_items)}</pre></div>'
    else:
        items_html = '<p class="no-open">No open items.</p>'

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>retinal_sim — Status Report</title>
  <style>{_STYLE}</style>
</head>
<body>
<h1>retinal_sim — Status Report</h1>
<p class="timestamp">Generated: {timestamp}</p>

<h2>Test Results</h2>
<div class="summary-box {box_cls}">{_escape(box_text)}</div>
{failures_html}

<h2>Phase Status</h2>
<table>
  <thead>
    <tr>
      <th>#</th><th>Component</th><th>Status</th><th>Tests</th><th>Notes</th>
    </tr>
  </thead>
  <tbody>
{phase_table_rows}  </tbody>
</table>

<h2>Validation Figures</h2>
{figures_html}

<h2>Open Code Review Items</h2>
{items_html}
</body>
</html>
"""


def _escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Running pytest...", flush=True)
    summary, failures, rc = run_pytest()
    counts = parse_pytest_summary(summary)
    print(f"  {summary}")

    print("Parsing PROGRESS.md...", flush=True)
    progress_text = (ROOT / "PROGRESS.md").read_text(encoding="utf-8")
    phase_rows = parse_phase_table(progress_text)

    print("Parsing CODEREVIEW.md...", flush=True)
    codereview_text = (ROOT / "CODEREVIEW.md").read_text(encoding="utf-8")
    open_items = parse_open_items(codereview_text)

    print("Generating validation figures...", flush=True)
    figure_uris: list[tuple[str, str]] = []
    try:
        import matplotlib.pyplot as plt
        fig1 = make_nomogram_figure()
        figure_uris.append(("Phase 1 — Govardovskii Nomogram", _fig_to_data_uri(fig1)))
        plt.close(fig1)
        fig2 = make_voronoi_figure()
        figure_uris.append(("Phase 8 — Voronoi Activation & Reconstruction", _fig_to_data_uri(fig2)))
        plt.close(fig2)
    except Exception as exc:
        print(f"  Warning: figure generation failed: {exc}", flush=True)

    print("Assembling HTML...", flush=True)
    html = build_html(phase_rows, summary, counts, failures, open_items, figure_uris)

    out = REPORTS_DIR / "status_latest.html"
    out.write_text(html, encoding="utf-8")
    print(f"\nReport written to: {out}")
    if rc != 0:
        sys.exit(rc)


if __name__ == "__main__":
    main()
