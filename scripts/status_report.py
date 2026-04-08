#!/usr/bin/env python
"""Generate reports/status_latest.html.

Produces a self-contained audit dashboard that distinguishes:
- automated test status,
- implementation phase status,
- architecture/validation coverage references,
- open code review findings, and
- documentation drift warnings.
"""
from __future__ import annotations

import base64
import io
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DOC_PHASE_PATTERNS = {
    "AGENTS.md": r"Phases?\s+1[\u2013\-]13\s+are\s+COMPLETE",
    "CLAUDE.md": r"Phase 13|Full validation report generator",
    "PROGRESS.md": r"Phases?\s+1[\u2013\-]13\s+are\s+complete|^\|\s*13\s*\|.+\*\*COMPLETE\*\*",
}


def run_pytest() -> tuple[str, str, int, str]:
    """Return (summary_line, failure_details, returncode, command_string)."""
    command = [sys.executable, "-m", "pytest", "--tb=short", "-q", "--no-header"]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    output = (result.stdout + result.stderr).strip()
    lines = output.splitlines()
    summary = next((line for line in reversed(lines) if line.strip()), "no output")

    failures: List[str] = []
    in_failure_block = False
    for line in lines:
        if re.match(r"^(FAILED |ERROR |={3,})", line):
            in_failure_block = True
        if in_failure_block:
            failures.append(line)

    return summary, "\n".join(failures), result.returncode, " ".join(command)


def parse_pytest_summary(summary: str) -> Dict[str, float]:
    counts: Dict[str, float] = {"passed": 0, "failed": 0, "skipped": 0, "error": 0, "duration": 0.0}
    for key in ("passed", "failed", "skipped", "error"):
        match = re.search(rf"(\d+) {key}", summary)
        if match:
            counts[key] = float(match.group(1))
    match = re.search(r"in ([\d.]+)s", summary)
    if match:
        counts["duration"] = float(match.group(1))
    return counts


def _parse_markdown_table(text: str, header_prefix: str) -> List[List[str]]:
    rows: List[List[str]] = []
    in_table = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(header_prefix):
            in_table = True
            continue
        if not in_table:
            continue
        if stripped.startswith("|---"):
            continue
        if stripped.startswith("|"):
            cells = [cell.strip() for cell in stripped.split("|")[1:-1]]
            if cells:
                rows.append([re.sub(r"\*\*(.+?)\*\*", r"\1", cell) for cell in cells])
        else:
            in_table = False
    return rows


def parse_phase_table(text: str) -> List[List[str]]:
    rows = _parse_markdown_table(text, "| Phase |")
    return [row for row in rows if row and re.match(r"^\d+$", row[0])]


def parse_milestone_table(text: str) -> List[List[str]]:
    return _parse_markdown_table(text, "| Milestone |")


def parse_open_items(text: str) -> str:
    match = re.search(r"## Open Items\n(.*?)(?=\n## |\Z)", text, re.DOTALL)
    if not match:
        return ""
    content = match.group(1).strip()
    return "" if content in ("*(none)*", "") else content


def detect_documentation_drift(docs: Dict[str, str]) -> List[str]:
    warnings: List[str] = []
    for name, pattern in DOC_PHASE_PATTERNS.items():
        text = docs.get(name, "")
        if not re.search(pattern, text, re.IGNORECASE):
            warnings.append(f"{name} does not clearly advertise the current Phase 13-complete state.")
    if "## Remediation Milestones" not in docs.get("PROGRESS.md", ""):
        warnings.append("PROGRESS.md does not include a remediation milestone snapshot.")
    if "Open code review items should be addressed before starting new phase work." in docs.get("AGENTS.md", ""):
        warnings.append("AGENTS.md still frames phase progression language even though the repo is now in audit/reporting mode.")
    return warnings


def get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _fig_to_data_uri(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()


def make_nomogram_figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from retinal_sim.retina.opsin import LAMBDA_MAX, build_sensitivity_curves

    wl = np.arange(380, 721, 1, dtype=float)
    colors = {"S_cone": "#5b5bd6", "M_cone": "#3aac3a", "L_cone": "#d94c4c", "rod": "#888888"}
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for ax, species in zip(axes, ["human", "dog", "cat"]):
        curves = build_sensitivity_curves(species, wl)
        for receptor in ["rod", "S_cone", "M_cone", "L_cone"]:
            if receptor not in curves:
                continue
            ax.plot(wl, curves[receptor], color=colors[receptor], lw=1.5, label=f"{receptor} ({LAMBDA_MAX[species][receptor]:.0f} nm)")
        ax.set_title(species.capitalize())
        ax.set_xlabel("Wavelength (nm)")
        ax.grid(True, lw=0.3, alpha=0.4)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("Relative sensitivity")
    fig.suptitle("Reference figure: Govardovskii nomogram")
    plt.tight_layout()
    return fig


def build_html(
    *,
    phase_rows: List[List[str]],
    milestone_rows: List[List[str]],
    pytest_summary: str,
    pytest_counts: Dict[str, float],
    pytest_failures: str,
    pytest_command: str,
    open_items: str,
    doc_warnings: List[str],
    validation_report_exists: bool,
    figure_uris: List[tuple[str, str]],
    git_commit: str,
) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    tests_ok = pytest_counts["failed"] == 0 and pytest_counts["error"] == 0
    box_cls = "all-pass" if tests_ok else "has-fail"

    phase_table_rows = "".join(
        "<tr>" + "".join(f"<td>{_escape(cell)}</td>" for cell in row) + "</tr>"
        for row in phase_rows
    )
    milestone_table_rows = "".join(
        "<tr>" + "".join(f"<td>{_escape(cell)}</td>" for cell in row) + "</tr>"
        for row in milestone_rows
    ) or (
        "<tr><td colspan=\"3\"><em>No remediation milestones recorded in PROGRESS.md.</em></td></tr>"
    )

    figure_html = "".join(
        f'<figure><img src="{uri}" alt="{_escape(caption)}"><figcaption>{_escape(caption)}</figcaption></figure>'
        for caption, uri in figure_uris
    ) or "<p><em>No validation figures generated.</em></p>"

    doc_warning_html = (
        "<ul>" + "".join(f"<li>{_escape(item)}</li>" for item in doc_warnings) + "</ul>"
        if doc_warnings else
        "<p>No documentation drift detected across the checked top-level docs.</p>"
    )

    open_items_html = (
        f"<pre>{_escape(open_items)}</pre>"
        if open_items else
        "<p>No open code review findings in CODEREVIEW.md.</p>"
    )

    failures_html = f"<pre>{_escape(pytest_failures)}</pre>" if pytest_failures else "<p>No failure excerpts.</p>"
    validation_link_note = (
        "Full validation report artifact detected in reports/validation_report.html."
        if validation_report_exists else
        "Full validation report artifact not found; status page references the validation system but no HTML artifact is present."
    )

    style = """
body { font-family: system-ui, -apple-system, sans-serif; max-width: 1020px; margin: 40px auto; padding: 0 24px; color: #222; line-height: 1.5; }
h1 { border-bottom: 2px solid #333; padding-bottom: 6px; }
h2 { border-bottom: 1px solid #ccc; margin-top: 2.2em; }
.summary-box { display: inline-block; padding: 10px 18px; border-radius: 6px; font-size: 1.05em; font-weight: bold; margin-bottom: 8px; }
.all-pass { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
.has-fail { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
.panel { background: #fafafa; border: 1px solid #ddd; border-radius: 6px; padding: 12px 14px; margin: 12px 0; }
.warning { background: #fff3cd; border-color: #ffe08a; }
table { border-collapse: collapse; width: 100%; margin-top: 8px; }
th, td { border: 1px solid #ddd; padding: 6px 10px; text-align: left; font-size: 0.92em; }
th { background: #f4f4f4; }
pre { background: #f6f6f6; border: 1px solid #ddd; padding: 12px 14px; border-radius: 4px; overflow-x: auto; white-space: pre-wrap; word-break: break-word; }
.figures { display: flex; flex-wrap: wrap; gap: 18px; }
.figures figure { flex: 1 1 420px; margin: 0; }
.figures img { max-width: 100%; border: 1px solid #e0e0e0; border-radius: 4px; }
"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>retinal_sim - Status Report</title>
  <style>{style}</style>
</head>
<body>
<h1>retinal_sim - Status Report</h1>
<p>Generated: {timestamp}</p>

<h2>Test Status</h2>
<div class="summary-box {box_cls}">{_escape(pytest_summary)}</div>
<div class="panel">
  <strong>Exact pytest command:</strong> <code>{_escape(pytest_command)}</code><br>
  <strong>Git commit:</strong> <code>{_escape(git_commit)}</code><br>
  <strong>Interpretation:</strong> This section is automated test status only. It is not the same thing as architecture coverage or code review status.
</div>
<h3>Failure Excerpts</h3>
{failures_html}

<h2>Remediation Milestones</h2>
<div class="panel">
  This table is sourced from <code>PROGRESS.md</code>. It tracks roadmap closure and claim-calibrated project status, not just phase completion.
</div>
<table>
  <thead><tr><th>Milestone</th><th>Status</th><th>Current evidence</th></tr></thead>
  <tbody>{milestone_table_rows}</tbody>
</table>

<h2>Implementation Phase Status</h2>
<div class="panel">
  This table is sourced from <code>PROGRESS.md</code>. It tracks implementation completion, not validation depth or audit completeness.
</div>
<table>
  <thead><tr><th>#</th><th>Component</th><th>Status</th><th>Tests</th><th>Notes</th></tr></thead>
  <tbody>{phase_table_rows}</tbody>
</table>

<h2>Validation and Architecture Audit Status</h2>
<div class="panel">
  <strong>Validation status:</strong> implemented validation checks are reported in the full validation audit artifact.<br>
  <strong>Architecture audit status:</strong> broader than test passing; it also depends on coverage gaps, deferred simplifications, and code review findings.<br>
  <strong>Artifact note:</strong> {_escape(validation_link_note)}
</div>

<h2>Documentation Drift Checks</h2>
<div class="panel warning">
  {doc_warning_html}
</div>

<h2>Open Code Review Findings</h2>
<div class="panel">
  {open_items_html}
</div>

<h2>Reference Validation Figures</h2>
<div class="figures">
  {figure_html}
</div>
</body>
</html>
"""


def main() -> None:
    print("Running pytest...", flush=True)
    summary, failures, rc, pytest_command = run_pytest()
    print(f"  {summary}")

    progress_text = (ROOT / "PROGRESS.md").read_text(encoding="utf-8")
    codereview_text = (ROOT / "CODEREVIEW.md").read_text(encoding="utf-8")
    docs = {
        "AGENTS.md": (ROOT / "AGENTS.md").read_text(encoding="utf-8"),
        "CLAUDE.md": (ROOT / "CLAUDE.md").read_text(encoding="utf-8"),
        "PROGRESS.md": progress_text,
    }

    phase_rows = parse_phase_table(progress_text)
    milestone_rows = parse_milestone_table(progress_text)
    open_items = parse_open_items(codereview_text)
    doc_warnings = detect_documentation_drift(docs)

    figure_uris: List[tuple[str, str]] = []
    try:
        import matplotlib.pyplot as plt

        fig = make_nomogram_figure()
        figure_uris.append(("Govardovskii nomogram reference", _fig_to_data_uri(fig)))
        plt.close(fig)
    except Exception as exc:
        print(f"  Warning: figure generation failed: {exc}", flush=True)

    html = build_html(
        phase_rows=phase_rows,
        milestone_rows=milestone_rows,
        pytest_summary=summary,
        pytest_counts=parse_pytest_summary(summary),
        pytest_failures=failures,
        pytest_command=pytest_command,
        open_items=open_items,
        doc_warnings=doc_warnings,
        validation_report_exists=(REPORTS_DIR / "validation_report.html").exists(),
        figure_uris=figure_uris,
        git_commit=get_git_commit(),
    )

    out = REPORTS_DIR / "status_latest.html"
    out.write_text(html, encoding="utf-8")
    print(f"Report written to: {out}")
    if rc != 0:
        sys.exit(rc)


if __name__ == "__main__":
    main()
