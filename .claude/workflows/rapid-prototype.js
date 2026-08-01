export const meta = {
  name: 'rapid-prototype',
  description: 'Scan → Fix → Test → Report cycle for OpenDike rapid prototyping',
  phases: [
    { title: 'Scan', detail: 'Find syntax, import, and architecture errors in all 9 source files (parallel)' },
    { title: 'Fix', detail: 'Auto-write fixes for critical errors directly to source files' },
    { title: 'Test', detail: 'Run local verification scripts via uv (skip LM-Studio-dependent ones)' },
    { title: 'Report', detail: 'Synthesize findings into a timestamped Markdown report in reports/' },
  ],
}

const PROJECT_ROOT = '/Users/lakhwinder/air/OpenDike'

const SOURCE_FILES = [
  'src/opendike/models.py',
  'src/opendike/memory.py',
  'src/opendike/experts.py',
  'src/opendike/wrapper.py',
  'src/opendike/learning.py',
  'src/opendike/train.py',
  'src/opendike/config.py',
  'src/opendike/core.py',
  'src/opendike/interactive_run.py',
]

const SCAN_SCHEMA = {
  type: 'object',
  properties: {
    file: { type: 'string' },
    is_clean: { type: 'boolean' },
    errors: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          error_type: { type: 'string' },
          description: { type: 'string' },
          line: { type: 'number' },
          severity: { type: 'string' },
          suggestion: { type: 'string' },
        },
        required: ['error_type', 'description', 'severity'],
      },
    },
    summary: { type: 'string' },
  },
  required: ['file', 'is_clean', 'errors', 'summary'],
}

const FIX_SCHEMA = {
  type: 'object',
  properties: {
    file: { type: 'string' },
    fixed: { type: 'boolean' },
    changes_made: { type: 'string' },
    remaining_issues: { type: 'string' },
  },
  required: ['file', 'fixed', 'changes_made'],
}

const TEST_SCHEMA = {
  type: 'object',
  properties: {
    script: { type: 'string' },
    status: { type: 'string' },
    output: { type: 'string' },
    error_message: { type: 'string' },
    notes: { type: 'string' },
  },
  required: ['script', 'status', 'output'],
}

// ── PHASE 1: SCAN ─────────────────────────────────────────────────────────
phase('Scan')

const scanResults = await parallel(SOURCE_FILES.map(file => () =>
  agent(
    `You are auditing a Python file in the OpenDike codebase for errors.

Read: ${PROJECT_ROOT}/${file}

Check for ALL of the following:

1. SYNTAX ERRORS — anything that would fail: python3 -m py_compile
2. IMPORT ERRORS — missing imports, wrong module paths, circular imports
3. OPENDIKE ARCHITECTURE VIOLATIONS:
   - MoralVector MUST have exactly 7 numpy dimensions in this fixed index order:
     [0]=care_harm  [1]=fairness_proportionality  [2]=loyalty_betrayal
     [3]=authority_subversion  [4]=sanctity_degradation  [5]=liberty_oppression
     [6]=deontological_vs_utilitarian
   - MemPalace storage structure: {layer_type: {layer_id: [MoralTrace]}}
   - Expert key format: "{layer_type}:{layer_id}" (e.g. "personal:user_123")
   - Config: always 'from opendike.config import config'; never instantiate Config() directly
4. RUNTIME ERRORS — undefined variables, wrong return types, missing required args

Severity:
  "critical" — causes crash or silent wrong behavior
  "warning"  — runtime risk under certain inputs
  "info"     — quality/style improvement

Return is_clean=true and empty errors array if the file has no issues.`,
    { label: `scan:${file.split('/').pop()}`, phase: 'Scan', schema: SCAN_SCHEMA, agentType: 'code-doctor' }
  )
))

const validScans = scanResults.filter(Boolean)
const allErrors = validScans.flatMap(r => r.errors.map(e => ({ ...e, file: r.file })))
const critical = allErrors.filter(e => e.severity === 'critical')
const warnings = allErrors.filter(e => e.severity === 'warning')

log(`Scan complete — critical: ${critical.length}, warnings: ${warnings.length}, info: ${allErrors.filter(e => e.severity === 'info').length}`)

// ── PHASE 2: FIX ──────────────────────────────────────────────────────────
phase('Fix')

let fixResults = []

if (critical.length > 0) {
  const byFile = {}
  for (const err of critical) {
    if (!byFile[err.file]) byFile[err.file] = []
    byFile[err.file].push(err)
  }

  fixResults = await parallel(Object.entries(byFile).map(([file, errors]) => () =>
    agent(
      `You are fixing critical Python errors in the OpenDike codebase.

File: ${PROJECT_ROOT}/${file}

Critical errors to fix:
${errors.map((e, i) => `${i + 1}. [${e.error_type}] ${e.line ? 'Line ' + e.line + ': ' : ''}${e.description}
   Suggestion: ${e.suggestion || 'Apply minimal fix'}`).join('\n\n')}

Rules:
- Read the file first with the Read tool
- Apply MINIMAL targeted fixes — only change what's listed above
- Do NOT refactor, rename, or change surrounding logic
- Use the Edit tool (not Write) so the diff is visible
- Preserve all OpenDike invariants:
  * MoralVector numpy index order: [0]=care_harm through [6]=deontological_vs_utilitarian
  * MemPalace.storage: {layer_type: {layer_id: [MoralTrace]}}
  * Expert key format: "{layer_type}:{layer_id}"
  * Config: always import singleton, never instantiate`,
      { label: `fix:${file.split('/').pop()}`, phase: 'Fix', schema: FIX_SCHEMA, agentType: 'code-doctor' }
    )
  ))

  const fixed = fixResults.filter(Boolean).filter(r => r.fixed).length
  log(`Fix complete — ${fixed}/${Object.keys(byFile).length} files fixed`)
} else {
  log('Fix — no critical errors, skipping')
}

// ── PHASE 3: TEST ─────────────────────────────────────────────────────────
phase('Test')

const TEST_TARGETS = [
  {
    name: 'core-mock-flow',
    script: 'src/opendike/core.py',
    note: 'Falls back to mock LLM if Gemma unavailable. Expected output includes "--- Initial Call ---".',
    requires_lm_studio: false,
    requires_docs: false,
  },
  {
    name: 'expert-embeddings',
    script: 'scripts/test_expert_embeddings.py',
    note: 'Requires docs/ dir and convert_docs_to_experts.py to have been run first.',
    requires_lm_studio: false,
    requires_docs: true,
  },
  {
    name: 'experts-loading',
    script: 'scripts/test_experts_loading.py',
    note: 'Requires convert_docs_to_experts.py to have been run first.',
    requires_lm_studio: false,
    requires_docs: true,
  },
  {
    name: 'lora-adaptation',
    script: 'scripts/test_lora_adaptation.py',
    note: 'Requires convert_docs_to_experts.py to have been run first.',
    requires_lm_studio: false,
    requires_docs: true,
  },
]

const testResults = await parallel(TEST_TARGETS.map(t => () =>
  agent(
    `Run a local test for the OpenDike project.

Test name: ${t.name}
Script: ${t.script}
Context: ${t.note}

${t.requires_docs ? `PREREQUISITE CHECK: Before running, verify that ${PROJECT_ROOT}/docs/ exists
and that ${PROJECT_ROOT}/data/memory_storage.json exists (populated by convert_docs_to_experts.py).
If either is missing, return status="skip" — do NOT treat it as a failure.` : ''}

Steps:
1. ${t.requires_docs ? 'Check prerequisites (ls docs/ and ls data/memory_storage.json)' : 'No prerequisite check needed'}
2. Run: cd ${PROJECT_ROOT} && uv run python ${t.script}
3. Capture full stdout and stderr
4. Classify result:
   - "pass"  — ran successfully with expected output
   - "fail"  — ran but wrong output or assertion failed
   - "skip"  — prerequisite missing (docs/, memory_storage.json)
   - "error" — unexpected crash unrelated to prerequisites`,
    { label: `test:${t.name}`, phase: 'Test', schema: TEST_SCHEMA, agentType: 'opendike-tester' }
  )
))

const passed = testResults.filter(Boolean).filter(r => r.status === 'pass').length
const failed = testResults.filter(Boolean).filter(r => r.status === 'fail').length
const skipped = testResults.filter(Boolean).filter(r => r.status === 'skip' || r.status === 'error').length

log(`Test complete — passed: ${passed}, failed: ${failed}, skipped/error: ${skipped}`)

// ── PHASE 4: REPORT ───────────────────────────────────────────────────────
phase('Report')

const reportData = {
  scan: validScans.map(r => ({ file: r.file, is_clean: r.is_clean, errors: r.errors, summary: r.summary })),
  fix: fixResults.filter(Boolean),
  test: testResults.filter(Boolean),
  totals: {
    files_scanned: SOURCE_FILES.length,
    critical: critical.length,
    warnings: warnings.length,
    files_fixed: fixResults.filter(Boolean).filter(r => r.fixed).length,
    tests_passed: passed,
    tests_failed: failed,
    tests_skipped: skipped,
  },
}

const reportPath = await agent(
  `Generate a prototype review report for the OpenDike project and write it to disk.

## Data to synthesize

${JSON.stringify(reportData, null, 2)}

## Report structure (write as Markdown)

### 1. Executive Summary
- One-paragraph health assessment (healthy / needs attention / critical issues)
- Stats table: Files Scanned | Critical Errors | Warnings | Files Fixed | Tests Passed | Tests Failed | Tests Skipped

### 2. Code Quality — Per-File Analysis
- Table: File | Status | Issue Count | Key Finding
- For files with errors: list each error with type, line, description, suggested fix

### 3. Fixes Applied
- For each fixed file: what changed and why
- Any errors that could not be auto-fixed and next steps

### 4. Test Results
- Table: Test | Status | Notes
- For failures/errors: root cause and how to resolve

### 5. Architectural Observations
- Patterns of recurring issues across files (e.g. MoralVector index misuse, wrong Config usage)
- OpenDike-specific risks in the current implementation

### 6. Next Steps
- P1 (critical — address now)
- P2 (important — address this sprint)
- P3 (nice-to-have)

## Output instructions
1. Run: date '+%Y%m%d-%H%M%S' to get a timestamp
2. Write the full report to: ${PROJECT_ROOT}/reports/prototype-<timestamp>.md
   (run: mkdir -p ${PROJECT_ROOT}/reports first)
3. Return only the absolute path to the written file`,
  { label: 'report', phase: 'Report' }
)

return {
  summary: reportData.totals,
  report_path: reportPath,
}
