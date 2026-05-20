from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("mapper-v21-decoder-evals")
    group.addoption(
        "--run-mapper-v21-decoder-evals",
        action="store_true",
        default=False,
        help="Run opt-in mapper v2.1 decoder profiler/eval experiments.",
    )
    group.addoption(
        "--mapper-v21-decoder-eval-output-dir",
        default=None,
        help="Directory for aggregate mapper v2.1 decoder eval JSON summaries.",
    )
    group.addoption(
        "--mapper-v21-decoder-eval-device",
        default="cpu",
        help="Torch device for mapper v2.1 decoder eval experiments.",
    )
    group.addoption(
        "--mapper-v21-decoder-eval-repeat",
        type=int,
        default=1,
        help="Measured repeat count for each mapper v2.1 decoder profiler scope.",
    )
    group.addoption(
        "--mapper-v21-decoder-eval-warmup",
        type=int,
        default=0,
        help="Warmup repeat count before each measured mapper v2.1 decoder profiler scope.",
    )
    group.addoption(
        "--mapper-v21-decoder-eval-prefix-lengths",
        default="1,2,4,8,16",
        help="Comma-separated prefix lengths for the mapper v2.1 decoder prefix sweep.",
    )
    group.addoption(
        "--mapper-v21-decoder-eval-skip-internal-grammar-mask",
        action="store_true",
        default=False,
        help="Run the mapper v2.1 prefix sweep with the model-internal grammar mask disabled.",
    )
    group.addoption(
        "--mapper-v21-decoder-eval-no-profiler",
        action="store_true",
        default=False,
        help="Disable torch.profiler while still running wall-clock mapper v2.1 decoder probes.",
    )
    group.addoption(
        "--mapper-v21-decoder-eval-rollout-ms",
        type=int,
        default=16_000,
        help="Synthetic full-rollout chart length in milliseconds for mapper v2.1 no-TS-penalty eval.",
    )
    group.addoption(
        "--mapper-v21-decoder-eval-rollout-max-tokens-per-window",
        type=int,
        default=256,
        help="Maximum generated sparse tokens per 8s window for mapper v2.1 rollout eval.",
    )
    group.addoption(
        "--mapper-v21-decoder-eval-render-reamber",
        action="store_true",
        default=False,
        help="Render the generated rollout .osu spans with Reamber when available.",
    )
    group.addoption(
        "--mapper-v21-decoder-checkpoint",
        default=None,
        help="Optional mapper v2.1 checkpoint to load once into the session-scoped eval state.",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "mapper_v21_decoder_eval: opt-in mapper v2.1 decoder profiler/eval experiment",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--run-mapper-v21-decoder-evals"):
        return
    skip_marker = pytest.mark.skip(reason="use --run-mapper-v21-decoder-evals to run mapper v2.1 decoder evals")
    for item in items:
        if item.get_closest_marker("mapper_v21_decoder_eval") is not None:
            item.add_marker(skip_marker)
