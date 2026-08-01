# OpenDike Rapid Prototype

Trigger: `/prototype`

## What this does

Runs the full OpenDike rapid-prototyping cycle using a 4-phase multi-agent workflow:

| Phase | Agents | What happens |
|-------|--------|-------------|
| **Scan** | 9 (parallel, one per source file) | Detect syntax, import, and architecture errors |
| **Fix** | N (parallel, one per file with critical errors) | Auto-write fixes directly to source files |
| **Test** | 4 (parallel) | Run local scripts via `uv run`; skip LM-Studio-dependent ones |
| **Report** | 1 | Synthesize all findings into a timestamped Markdown file in `reports/` |

## How to invoke this skill

Call the Workflow tool with the project workflow script:

```
Workflow({ scriptPath: '/Users/lakhwinder/air/OpenDike/.claude/workflows/rapid-prototype.js' })
```

## After the workflow completes

1. Print a short summary paragraph: files scanned, issues found by severity, fixes applied, test pass/fail/skip
2. Show the full path to the generated report
3. If any critical errors remain unfixed (fix phase returned `fixed: false`), list them with file + description
4. Offer to re-run with `args: { fix_only: true }` if there are remaining unfixed issues

## Optional args (passed via `args` object)

- `{ scan_only: true }` — skip Fix and Test phases, only scan and report
- `{ no_fix: true }` — skip Fix phase, still run tests
- `{ verbose: true }` — include full script output in the report (not just summaries)
