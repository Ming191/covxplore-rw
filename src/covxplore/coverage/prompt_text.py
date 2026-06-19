NO_TESTS_WITH_MCDC = (
    "No tests executed yet. Target is {total_mcdc_pairs} MC/DC pairs. "
    "Do NOT stop. Generate and execute a first compilable test case."
)
NO_TESTS_WITHOUT_MCDC = (
    "No tests executed yet. Execute a first compilable test case to assess "
    "statement and branch coverage."
)
COMPILE_ERROR = (
    "Last execution status is COMPILE_ERROR. Coverage is not complete. "
    "Do NOT stop. Fix compilation issues and re-run execute_testcase."
)
FAILED = (
    "Last execution status is FAILED. Coverage is not complete. "
    "Inspect execute_log and continue with the next test."
)
UNKNOWN = (
    "Last execution status is UNKNOWN. Coverage state may be incomplete. "
    "Do NOT stop. Re-run with a valid test and continue."
)
SUCCESS_PREFIX = "SUCCESS! All coverage targets are now met (statement, branch"
SUCCESS_SUFFIX = (
    "). DO NOT call any more tools. "
    "Please output your final 'DONE: ...' message immediately to finish the task."
)
CONDITION_IDENTITY_NOTE = (
    "Condition identity note: MC/DC condition identity is nodeId; "
    "condition text may repeat across different nodeIds."
)
UNCOVERED_MCDC_HEADER = "The following MC/DC condition polarities are NOT yet covered:"
STUCK_ALL_HEADER = "All remaining MC/DC obligations are likely stuck/unobservable from backend feedback."
STUCK_ALL_ADVICE = "Do NOT continue issuing near-duplicate tests for these."
STUCK_REPORT_HEADER = "Report likely instrumentation gap for the remaining nodeId/polarities:"
STUCK_RATIONALE = "Rationale: opposite polarity was observed repeatedly, but this polarity never appears."
STUCK_PARTIAL_HEADER = "Potentially stuck obligations (opposite polarity seen repeatedly):"
STUCK_PARTIAL_ADVICE = "  Do not fixate on these first; cover other obligations, then retry with a different baseline/path."
REDUNDANT_WARNING_3 = (
    "\nWARNING: 3+ consecutive redundant tests (0 new MC/DC pairs). "
    "You are likely stuck at a local optimum. "
    "Do NOT keep issuing near-duplicate tests. "
    "Switch to a different nodeId family/branch structure. "
    "If the next attempt is still redundant, output final DONE summary immediately to stop token waste."
)
REDUNDANT_CAUTION_2 = (
    "\nCAUTION: 2 consecutive redundant tests. Switch to a different baseline passing test and "
    "drive a different code path for the targeted nodeId/polarity."
)
TARGETING_ADVICE = (
    "\nTarget the test path that satisfies the highest number of conditions. "
    "Prioritize paths that cover multiple uncovered conditions simultaneously. "
    "If multiple paths are possible, choose the one that increases overall condition coverage the most. "
    "Prefer modifying an existing passing test when possible, but allow generating a new test if needed."
)
MAX_STATEMENT_LINES = 15
MAX_BRANCH_LINES = 15
MAX_STUCK_LINES = 8
MAX_PARTIAL_STUCK_LINES = 5
