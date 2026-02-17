"""
Pytest configuration for Lab 2 hidden tests.
Loads variant configuration and provides fixtures for variant-aware testing.
"""

import json
import sys
import pytest
from pathlib import Path

# Ensure src/ is importable
SRC_DIR = Path(__file__).parent.parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


@pytest.fixture(scope="session")
def variant_config():
    """Load student's variant configuration from .variant_config.json or get_variant.py."""
    config_path = Path(__file__).parent.parent.parent / ".variant_config.json"
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)
    # Fall back to generating via get_variant.py
    scripts_dir = Path(__file__).parent.parent.parent / "scripts"
    sys.path.insert(0, str(scripts_dir))
    try:
        from get_variant import get_my_variant
        return get_my_variant()
    finally:
        sys.path.pop(0)


@pytest.fixture
def grade_thresholds(variant_config):
    """Return grade threshold values from the variant config."""
    return variant_config["parameters"]["grade_thresholds"]


@pytest.fixture
def test_samples(variant_config):
    """Return test sample values from the variant config."""
    return variant_config["parameters"]["test_samples"]


@pytest.fixture
def base_rate(variant_config):
    """Return base drilling rate from the variant config."""
    return variant_config["parameters"]["base_rate"]


@pytest.fixture
def drilling_depths(variant_config):
    """Return drilling depth test values from the variant config."""
    return variant_config["parameters"]["drilling_depths"]


@pytest.fixture
def alternative_samples():
    """Return alternative test data to catch hardcoded values.

    These values are deliberately different from any default or
    commonly hardcoded sample sets.
    """
    return [0.3, 1.7, 2.9, 4.1, 0.0, 3.6, 1.2, 2.4]


@pytest.fixture(autouse=True)
def cleanup_plots():
    """Close any matplotlib figures after each test to prevent resource leaks."""
    yield
    try:
        import matplotlib.pyplot as plt
        plt.close('all')
    except ImportError:
        pass
