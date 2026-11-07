from diarize_nemotron import verify_coverage

verify_coverage(30.0, 0.01, 3000, 3001)
for valid, output in [(2000, 2000), (3001, 2999)]:
    try:
        verify_coverage(30.0, 0.01, valid, output)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Incomplete coverage was accepted")
print("coverage checks passed")
