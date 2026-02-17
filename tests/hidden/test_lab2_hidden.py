"""
Lab 2 Hidden Tests - Control Flow & Functions
These tests verify correctness against each student's unique variant parameters.
Students do NOT see these tests; they run during autograding only.
"""

import sys
from pathlib import Path

SRC_DIR = Path(__file__).parent.parent.parent / "src"


def _ensure_src():
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))


# ============================================================================
# Hidden Classification Tests
# ============================================================================

class TestHiddenClassification:
    """Hidden tests for classify_grade using variant thresholds."""

    def test_classify_at_high_threshold(self, grade_thresholds):
        """Grade exactly at high threshold should be High Grade."""
        _ensure_src()
        from lab2_classify import classify_grade
        result = classify_grade(grade_thresholds['high'])
        assert result is not None, "classify_grade returned None"
        assert "high" in result.lower(), \
            f"Grade {grade_thresholds['high']} (high threshold) should be 'High Grade', got '{result}'"

    def test_classify_above_high_threshold(self, grade_thresholds):
        """Grade well above high threshold should be High Grade."""
        _ensure_src()
        from lab2_classify import classify_grade
        test_val = grade_thresholds['high'] + 1.0
        result = classify_grade(test_val)
        assert result is not None
        assert "high" in result.lower(), f"Grade {test_val} should be 'High Grade', got '{result}'"

    def test_classify_at_medium_threshold(self, grade_thresholds):
        """Grade exactly at medium threshold should be Medium Grade."""
        _ensure_src()
        from lab2_classify import classify_grade
        result = classify_grade(grade_thresholds['medium'])
        assert result is not None
        assert "medium" in result.lower(), \
            f"Grade {grade_thresholds['medium']} (medium threshold) should be 'Medium Grade', got '{result}'"

    def test_classify_at_low_threshold(self, grade_thresholds):
        """Grade exactly at low threshold should be Low Grade."""
        _ensure_src()
        from lab2_classify import classify_grade
        result = classify_grade(grade_thresholds['low'])
        assert result is not None
        assert "low" in result.lower(), \
            f"Grade {grade_thresholds['low']} (low threshold) should be 'Low Grade', got '{result}'"

    def test_classify_just_below_low_threshold(self, grade_thresholds):
        """Grade just below low threshold should be Sub-economic."""
        _ensure_src()
        from lab2_classify import classify_grade
        test_val = grade_thresholds['low'] - 0.1
        if test_val >= 0:
            result = classify_grade(test_val)
            assert result is not None
            assert "sub" in result.lower(), \
                f"Grade {test_val} (below low threshold) should be 'Sub-economic', got '{result}'"

    def test_classify_with_alternative_values(self, grade_thresholds, alternative_samples):
        """Classify alternative samples to catch hardcoded logic."""
        _ensure_src()
        from lab2_classify import classify_grade
        for sample in alternative_samples:
            result = classify_grade(sample)
            assert result is not None, f"classify_grade({sample}) returned None"
            # Verify the classification is one of the valid categories
            result_lower = result.lower()
            valid = any(cat in result_lower for cat in ['high', 'medium', 'low', 'sub', 'invalid'])
            assert valid, f"classify_grade({sample}) returned unexpected '{result}'"


# ============================================================================
# Hidden Processing Tests
# ============================================================================

class TestHiddenProcessing:
    """Hidden tests for processing functions using variant data."""

    def test_process_samples_correct_count(self, test_samples):
        """process_samples should count only non-negative samples."""
        _ensure_src()
        from lab2_processing import process_samples
        result = process_samples(test_samples)
        assert result is not None, "process_samples returned None"
        expected_count = sum(1 for s in test_samples if s >= 0)
        assert result['count'] == expected_count, \
            f"Expected count {expected_count}, got {result['count']}"

    def test_process_samples_correct_mean(self, test_samples):
        """process_samples average should match expected calculation."""
        _ensure_src()
        from lab2_processing import process_samples
        result = process_samples(test_samples)
        assert result is not None
        valid = [s for s in test_samples if s >= 0]
        if valid:
            expected_avg = sum(valid) / len(valid)
            assert abs(result['average'] - expected_avg) < 0.01, \
                f"Expected average {expected_avg:.4f}, got {result['average']}"

    def test_process_samples_min_max(self, test_samples):
        """process_samples should report correct min and max of valid samples."""
        _ensure_src()
        from lab2_processing import process_samples
        result = process_samples(test_samples)
        assert result is not None
        valid = [s for s in test_samples if s >= 0]
        if valid:
            assert abs(result['max_grade'] - max(valid)) < 0.01, \
                f"Expected max {max(valid)}, got {result['max_grade']}"
            assert abs(result['min_grade'] - min(valid)) < 0.01, \
                f"Expected min {min(valid)}, got {result['min_grade']}"

    def test_count_by_category_with_variant(self, test_samples, grade_thresholds):
        """count_by_category totals should equal len(test_samples)."""
        _ensure_src()
        from lab2_processing import count_by_category
        result = count_by_category(test_samples, grade_thresholds)
        assert result is not None, "count_by_category returned None"
        total = sum(result.values())
        assert total == len(test_samples), \
            f"Category counts total ({total}) should equal sample count ({len(test_samples)})"

    def test_filter_samples_with_variant(self, test_samples, grade_thresholds):
        """filter_samples should correctly filter variant test data."""
        _ensure_src()
        from lab2_processing import filter_samples
        min_g = grade_thresholds['low']
        result = filter_samples(test_samples, min_grade=min_g)
        assert result is not None, "filter_samples returned None"
        for val in result:
            assert val >= min_g, f"Filtered value {val} is below min_grade {min_g}"


# ============================================================================
# Hidden Calculator Tests
# ============================================================================

class TestHiddenCalculator:
    """Hidden tests for drilling cost calculator using variant base_rate."""

    def test_drilling_cost_tier1_exact(self, base_rate):
        """Tier 1 only: cost for exactly 200m."""
        _ensure_src()
        from lab2_calculator import calculate_drilling_cost
        cost = calculate_drilling_cost(200, base_rate, 'medium')
        expected = 200 * base_rate
        assert cost is not None
        assert abs(cost - expected) < 0.01, f"200m cost: expected {expected}, got {cost}"

    def test_drilling_cost_tier2_boundary(self, base_rate):
        """Tier 2 boundary: cost for exactly 500m."""
        _ensure_src()
        from lab2_calculator import calculate_drilling_cost
        cost = calculate_drilling_cost(500, base_rate, 'medium')
        expected = (200 * base_rate) + (300 * base_rate * 1.25)
        assert cost is not None
        assert abs(cost - expected) < 0.01, f"500m cost: expected {expected}, got {cost}"

    def test_drilling_cost_with_variant_depths(self, drilling_depths, base_rate):
        """Calculate cost for each variant depth without error."""
        _ensure_src()
        from lab2_calculator import calculate_drilling_cost
        for depth in drilling_depths:
            cost = calculate_drilling_cost(depth, base_rate, 'medium')
            assert cost is not None and cost > 0, \
                f"Cost for depth {depth} should be positive, got {cost}"

    def test_drilling_cost_hardness_ratios(self, base_rate):
        """Hardness adjustments should produce correct ratios."""
        _ensure_src()
        from lab2_calculator import calculate_drilling_cost
        depth = 400
        cost_soft = calculate_drilling_cost(depth, base_rate, 'soft')
        cost_medium = calculate_drilling_cost(depth, base_rate, 'medium')
        cost_hard = calculate_drilling_cost(depth, base_rate, 'hard')
        assert cost_soft is not None and cost_medium is not None and cost_hard is not None
        assert abs(cost_soft / cost_medium - 0.9) < 0.01, "Soft/medium ratio should be 0.9"
        assert abs(cost_hard / cost_medium - 1.2) < 0.01, "Hard/medium ratio should be 1.2"

    def test_cost_breakdown_consistent_total(self, base_rate):
        """Breakdown subtotal should equal sum of tier costs."""
        _ensure_src()
        from lab2_calculator import calculate_cost_breakdown
        breakdown = calculate_cost_breakdown(750, base_rate)
        if breakdown is not None:
            tier_sum = breakdown['tier1_cost'] + breakdown['tier2_cost'] + breakdown['tier3_cost']
            assert abs(breakdown['subtotal'] - tier_sum) < 0.01, \
                f"Subtotal ({breakdown['subtotal']}) should equal tier sum ({tier_sum})"

    def test_format_cost_report_contains_hardness(self, base_rate):
        """format_cost_report should include the hardness value."""
        _ensure_src()
        from lab2_calculator import format_cost_report
        result = format_cost_report(500, 25000.0, 'hard', base_rate)
        if result is not None:
            assert 'hard' in result.lower(), "Report should contain the hardness value"


# ============================================================================
# Hidden Statistics Tests
# ============================================================================

class TestHiddenStatistics:
    """Hidden tests for statistics functions using variant data."""

    def test_generate_report_returns_string(self, test_samples, grade_thresholds):
        """generate_report should return a string."""
        _ensure_src()
        from lab2_statistics import generate_report
        result = generate_report(test_samples, grade_thresholds)
        assert result is not None, "generate_report returned None"
        assert isinstance(result, str), "generate_report should return a string"

    def test_generate_report_contains_statistics(self, test_samples, grade_thresholds):
        """generate_report should contain key statistical information."""
        _ensure_src()
        from lab2_statistics import generate_report
        result = generate_report(test_samples, grade_thresholds)
        if result is not None:
            valid = [s for s in test_samples if s >= 0]
            if valid:
                # Report should contain the count of valid samples
                assert str(len(valid)) in result, \
                    f"Report should contain valid sample count ({len(valid)})"

    def test_find_anomalies_with_variant(self, test_samples):
        """find_anomalies on variant data should return only values above threshold."""
        _ensure_src()
        from lab2_statistics import find_anomalies
        result = find_anomalies(test_samples, threshold_multiplier=2.0)
        if result is not None:
            valid = [s for s in test_samples if s >= 0]
            if valid:
                avg = sum(valid) / len(valid)
                threshold = avg * 2.0
                for val in result:
                    assert val > threshold, \
                        f"Anomaly {val} should be above threshold {threshold:.2f}"

    def test_find_anomalies_sorted_descending(self, test_samples):
        """find_anomalies should return values sorted in descending order."""
        _ensure_src()
        from lab2_statistics import find_anomalies
        result = find_anomalies(test_samples, threshold_multiplier=1.5)
        if result is not None and len(result) > 1:
            for i in range(len(result) - 1):
                assert result[i] >= result[i + 1], \
                    f"Anomalies should be sorted descending: {result}"

    def test_grade_distribution_with_variant(self, test_samples):
        """get_grade_distribution with variant data should produce correct bin count."""
        _ensure_src()
        from lab2_statistics import get_grade_distribution
        result = get_grade_distribution(test_samples, num_bins=5, grade_range=(0, 5))
        if result is not None:
            assert len(result) == 5, f"Expected 5 bins, got {len(result)}"
            # Total should equal the number of samples in the range [0, 5]
            in_range = [s for s in test_samples if 0 <= s <= 5]
            total = sum(result.values())
            assert total == len(in_range), \
                f"Total bin count ({total}) should equal samples in range ({len(in_range)})"


# ============================================================================
# Hidden Variant Verification Tests
# ============================================================================

class TestHiddenVariantVerification:
    """Verify that the student's variant configuration is valid."""

    def test_variant_has_required_keys(self, variant_config):
        """Variant config should have all required parameter keys."""
        params = variant_config.get("parameters", {})
        required = ['grade_thresholds', 'test_samples', 'drilling_depths', 'base_rate']
        for key in required:
            assert key in params, f"Variant config missing required key: {key}"

    def test_grade_thresholds_high_in_range(self, grade_thresholds):
        """High threshold should be in range 2.8 - 3.2."""
        high = grade_thresholds['high']
        assert 2.8 <= high <= 3.2, \
            f"High threshold ({high}) should be between 2.8 and 3.2"

    def test_base_rate_in_valid_set(self, base_rate):
        """Base rate should be one of the valid options."""
        valid_rates = [45, 50, 55, 60, 65, 70, 75]
        assert base_rate in valid_rates, \
            f"Base rate ({base_rate}) should be one of {valid_rates}"

    def test_test_samples_has_8_values(self, test_samples):
        """test_samples should have exactly 8 values."""
        assert len(test_samples) == 8, \
            f"test_samples should have 8 values, got {len(test_samples)}"

    def test_drilling_depths_has_4_values(self, drilling_depths):
        """drilling_depths should have exactly 4 values."""
        assert len(drilling_depths) == 4, \
            f"drilling_depths should have 4 values, got {len(drilling_depths)}"


# ============================================================================
# Hidden Integration Tests
# ============================================================================

class TestHiddenIntegration:
    """Integration tests combining multiple modules with variant data."""

    def test_classify_all_samples_matches_counts(self, test_samples, grade_thresholds):
        """Classifying all samples individually should match count_by_category totals."""
        _ensure_src()
        from lab2_classify import classify_grade
        from lab2_processing import count_by_category

        # Classify each sample individually
        individual_counts = {'high': 0, 'medium': 0, 'low': 0, 'subeconomic': 0, 'invalid': 0}
        for sample in test_samples:
            result = classify_grade(sample)
            if result is not None:
                r = result.lower()
                if 'high' in r and 'medium' not in r:
                    individual_counts['high'] += 1
                elif 'medium' in r:
                    individual_counts['medium'] += 1
                elif 'low' in r:
                    individual_counts['low'] += 1
                elif 'sub' in r:
                    individual_counts['subeconomic'] += 1
                elif 'invalid' in r:
                    individual_counts['invalid'] += 1

        # Compare with count_by_category
        category_counts = count_by_category(test_samples, grade_thresholds)
        if category_counts is not None:
            for key in ['high', 'medium', 'low', 'subeconomic', 'invalid']:
                assert individual_counts[key] == category_counts.get(key, 0), \
                    f"Category '{key}': individual={individual_counts[key]}, " \
                    f"count_by_category={category_counts.get(key, 0)}"

    def test_drilling_costs_all_variant_depths(self, drilling_depths, base_rate):
        """Calculate drilling costs for all variant depths across all hardness levels."""
        _ensure_src()
        from lab2_calculator import calculate_drilling_cost
        for depth in drilling_depths:
            costs = {}
            for hardness in ['soft', 'medium', 'hard']:
                cost = calculate_drilling_cost(depth, base_rate, hardness)
                assert cost is not None and cost > 0, \
                    f"Cost for depth={depth}, hardness='{hardness}' should be positive"
                costs[hardness] = cost
            # Verify ordering: soft < medium < hard
            assert costs['soft'] < costs['medium'] < costs['hard'], \
                f"For depth {depth}: soft ({costs['soft']}) < medium ({costs['medium']}) < hard ({costs['hard']}) expected"

    def test_full_pipeline(self, test_samples, grade_thresholds):
        """Full pipeline: process samples, then generate report."""
        _ensure_src()
        from lab2_processing import process_samples
        from lab2_statistics import generate_report

        # Step 1: Process samples
        stats = process_samples(test_samples)
        assert stats is not None, "process_samples returned None"
        assert stats['count'] > 0, "Should have at least one valid sample"

        # Step 2: Generate report
        report = generate_report(test_samples, grade_thresholds)
        if report is not None:
            assert isinstance(report, str), "Report should be a string"
            assert len(report) > 0, "Report should not be empty"
