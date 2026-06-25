def compute_accuracy(results: list[dict]) -> float:
    correct = sum(1 for r in results if r["prediction"] == r["label"])
    return correct / len(results) if results else 0.0
