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
    "Inspect execute_log, then re-run with a valid test and continue."
)
SUCCESS_PREFIX = "SUCCESS! All coverage targets are now met (statement, branch"
SUCCESS_SUFFIX = (
    "). DO NOT call any more tools. "
    "Please output your final 'DONE: ...' message immediately to finish the task."
)
MAX_STATEMENT_LINES = 15
MAX_BRANCH_LINES = 15
