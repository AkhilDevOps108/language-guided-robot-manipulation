from gripground.evaluation.metrics import EpisodeResult, action_mse, summarize_episode_results


def test_metrics_summary() -> None:
    results = [
        EpisodeResult(success=True, steps=40, latency_ms=2.0, failure_category="none"),
        EpisodeResult(success=False, steps=120, latency_ms=3.0, failure_category="timeout"),
    ]
    summary = summarize_episode_results(results)
    assert summary["episodes"] == 2
    assert summary["success_rate"] == 0.5
    assert summary["failure_categories"]["timeout"] == 1


def test_action_mse() -> None:
    import numpy as np

    pred = np.array([[0.0, 1.0], [1.0, 0.0]])
    target = np.array([[0.0, 0.0], [0.0, 0.0]])
    assert action_mse(pred, target) == 0.5

