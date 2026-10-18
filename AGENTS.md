## Agent skills

Process guides for coding agents live in `.agents/skills/`: a plain-Markdown copy of mattpocock-skills v1.3.1 that any harness can read. When a ticket says to build test-first, follow `.agents/skills/tdd/SKILL.md`. When it says to self-review, follow `.agents/skills/code-review/SKILL.md`. To work a whole ticket, follow `.agents/skills/implement/SKILL.md`. If a skill tells you to call a "Skill tool" your harness doesn't have, open that skill's `SKILL.md` and follow it directly.

### Issue tracker

Issues live in this repo's GitHub Issues (via the `gh` CLI). See `docs/agents/issue-tracker.md`.

### Triage labels

Default labels: needs-triage, needs-info, ready-for-agent, ready-for-human, wontfix. See `docs/agents/triage-labels.md`. A `ready-for-human` ticket is an owner step (it needs the trained models, the private manuscript, or a tag push). Don't attempt it; if your ticket is blocked by one, stop and say so.

### Domain docs

Single-context: the glossary is `CONTEXT.md` at the repo root (not `GLOSSARY.md`), and decisions live in `docs/adr/`. Read ADR-0001 before touching anything under `data/`. See `docs/agents/domain.md`.

## Working here

- Python via `uv`: `uv sync --no-group notebook`, then `uv run python -m pytest -q`. The test suite runs from a bare clone against the committed `data/processed/oos_forecasts.parquet`. The licensed raw inputs and the trained models aren't distributed and must never be needed by tests or the showcase export.
- Every filesystem path goes through the constants in `betaforecast/paths.py`.
