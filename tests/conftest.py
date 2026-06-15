def pytest_runtest_logreport(report):
    """
    Print an explicit outcome line for each test call.
    """
    if report.when != "call":
        return

    if report.passed:
        outcome = "PASSED"
    elif report.failed:
        outcome = "FAILED"
    elif report.skipped:
        outcome = "SKIPPED"
    else:
        outcome = report.outcome.upper()

    print(f"[pytest] {report.nodeid}:  \n{outcome}")
