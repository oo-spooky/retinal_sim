# LLM Coordination

Shared workflow for Codex and Claude. This file is the source of truth for repo-level coordination, touchlog usage, and markdown hygiene.

## Session Workflow

1. Read `TOUCHLOG.md` before starting.
2. Read `PROGRESS.md` for the live repo snapshot, `CODEREVIEW.md` for open findings, and `SCRATCHPAD.md` for unresolved short-lived notes.
3. Read your agent-specific bootstrap doc (`AGENTS.md` or `CLAUDE.md`) for tool-specific guidance.
4. For structural, scientific-claim, reporting, or validation work, also read `retinal_sim_architecture.md`, `docs/reporting_transparency.md`, and the dated `reports/*.md` artifacts relevant to the area you are touching.
5. Before ending the session, update any active docs whose source of truth changed, then append one touchlog entry.

## Active Documents

- `AGENTS.md`: Codex bootstrap and repo technical brief.
- `CLAUDE.md`: Claude bootstrap, orchestration notes, and Codex-from-Claude usage.
- `TOUCHLOG.md`: shared end-of-session handoff log.
- `PROGRESS.md`: current repo snapshot only.
- `SCRATCHPAD.md`: unresolved short-lived notes only.
- `CODEREVIEW.md`: open findings only.
- `retinal_sim_architecture.md`: canonical software architecture.
- `docs/reporting_transparency.md`: report wording and transparency contract.
- `reports/*.md`: dated audit, roadmap, and other reference artifacts.

## Touchlog Rules

- Append exactly one entry per working session, including review-only sessions.
- Use local time and append in chronological order.
- If no files changed, record `Changed/Reviewed: none` or list the files reviewed in read-only mode.
- Keep entries short and handoff-oriented.

Use this format:

```text
YYYY-MM-DD HH:MM MDT | Agent | session label
Scope: ...
Changed/Reviewed: ...
Outcome: ...
Handoff: ...
```

- Archive the oldest entries to `docs/archive/TOUCHLOG_ARCHIVE.md` when `TOUCHLOG.md` exceeds 25 entries or about 300 lines. Keep the newest 25 entries in the root file.

## Document Placement Rules

- Do not create a new root markdown file unless it has a durable, repo-wide role.
- Historical, resolved, or session-local material goes in `docs/archive/` or `TOUCHLOG.md`.
- Durable technical guidance belongs in `AGENTS.md`, `CLAUDE.md`, `retinal_sim_architecture.md`, `docs/reporting_transparency.md`, or dated `reports/*.md`, whichever is the true owner.
- `SCRATCHPAD.md` is not a permanent notebook. Promote durable notes out of it and clear it when items are resolved.
- `PROGRESS.md` is not a changelog. Keep it as a concise current-state snapshot.
- `CODEREVIEW.md` must stay open-items only. Move resolved material into the resolved archive.
- Dated `reports/*.md` files remain in `reports/`; they are not folded into `TOUCHLOG.md` or `docs/archive/` unless replaced by a newer dated report of the same type.

## Archive Policy

- Preserve history by moving it, not deleting it.
- Use dated archive filenames when retiring active markdown files.
- Prefer a single archive file per document family until it becomes too large, then start a new dated archive file.
