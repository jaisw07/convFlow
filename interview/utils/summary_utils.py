def summary_trigger(summary: str):

    # Approximate token check
    words = len(summary.split())

    # Count turns
    turns = summary.count("Q:")

    if turns >= 3:
        return True

    if words >= 1000:
        return True

    return False