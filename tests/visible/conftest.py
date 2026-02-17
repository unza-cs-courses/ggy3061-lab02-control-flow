"""
Pytest configuration for Lab 2 visible tests.
Loads variant configuration if available.
"""

import json
import pytest
from pathlib import Path


@pytest.fixture(scope="session")
def variant_config():
    """Load student's variant configuration."""
    config_path = Path(__file__).parent.parent.parent / ".variant_config.json"
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)
    # Default values for testing without variant config
    return {
        "parameters": {
            "grade_thresholds": {"high": 3.0, "medium": 2.0, "low": 1.0},
            "test_samples": [3.5, 2.1, 1.5, 0.8, -0.5, 4.2, 0.0, 1.9],
            "drilling_depths": [150, 350, 600, 800],
            "base_rate": 50
        }
    }


@pytest.fixture
def grade_thresholds(variant_config):
    """Return grade threshold values."""
    return variant_config["parameters"]["grade_thresholds"]


@pytest.fixture
def test_samples(variant_config):
    """Return test sample values."""
    return variant_config["parameters"]["test_samples"]


@pytest.fixture
def base_rate(variant_config):
    """Return base drilling rate."""
    return variant_config["parameters"]["base_rate"]


@pytest.fixture(autouse=True)
def cleanup_plots():
    """Close any matplotlib figures after each test to prevent resource leaks."""
    yield
    try:
        import matplotlib.pyplot as plt
        plt.close('all')
    except ImportError:
        pass
