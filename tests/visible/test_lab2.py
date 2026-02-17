"""
Lab 2 Visible Tests - Control Flow & Functions
These tests verify core functionality of control structures and functions.
"""

import subprocess
import sys
from pathlib import Path

SRC_DIR = Path(__file__).parent.parent.parent / "src"


def run_script(script_name, input_data=None):
    """Run a Python script and capture output."""
    script_path = SRC_DIR / script_name
    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
        input=input_data,
        timeout=10
    )
    return result


# ============================================================================
# Task 1: Grade Classification Tests
# ============================================================================

class TestTask1ClassifyGrade:
    """Tests for Task 1: classify_grade function"""

    def test_classify_file_exists(self):
        """lab2_classify.py file should exist"""
        assert (SRC_DIR / "lab2_classify.py").exists(), "lab2_classify.py not found in src/"

    def test_classify_runs_without_error(self):
        """lab2_classify.py should run without errors"""
        result = run_script("lab2_classify.py")
        assert result.returncode == 0, f"Script failed with error: {result.stderr}"

    def test_classify_function_exists(self):
        """classify_grade function should be defined"""
        sys.path.insert(0, str(SRC_DIR))
        try:
            from lab2_classify import classify_grade
            assert callable(classify_grade), "classify_grade should be a function"
        finally:
            sys.path.pop(0)

    def test_classify_high_grade(self, grade_thresholds):
        """Should classify high grades correctly"""
        sys.path.insert(0, str(SRC_DIR))
        try:
            from lab2_classify import classify_grade
            high_threshold = grade_thresholds['high']
            result = classify_grade(high_threshold + 0.5)
            assert result is not None, "classify_grade should return a value"
            assert "high" in result.lower(), f"Grade {high_threshold + 0.5} should be classified as High Grade"
        finally:
            sys.path.pop(0)

    def test_classify_medium_grade(self, grade_thresholds):
        """Should classify medium grades correctly"""
        sys.path.insert(0, str(SRC_DIR))
        try:
            from lab2_classify import classify_grade
            # Grade between medium and high thresholds
            test_grade = (grade_thresholds['medium'] + grade_thresholds['high']) / 2
            result = classify_grade(test_grade)
            assert result is not None, "classify_grade should return a value"
            assert "medium" in result.lower(), f"Grade {test_grade} should be Medium Grade"
        finally:
            sys.path.pop(0)

    def test_classify_low_grade(self, grade_thresholds):
        """Should classify low grades correctly"""
        sys.path.insert(0, str(SRC_DIR))
        try:
            from lab2_classify import classify_grade
            # Grade between low and medium thresholds
            test_grade = (grade_thresholds['low'] + grade_thresholds['medium']) / 2
            result = classify_grade(test_grade)
            assert result is not None, "classify_grade should return a value"
            assert "low" in result.lower(), f"Grade {test_grade} should be Low Grade"
        finally:
            sys.path.pop(0)

    def test_classify_subeconomic(self, grade_thresholds):
        """Should classify sub-economic grades correctly"""
        sys.path.insert(0, str(SRC_DIR))
        try:
            from lab2_classify import classify_grade
            # Grade between 0 and low threshold
            test_grade = grade_thresholds['low'] / 2
            result = classify_grade(test_grade)
            assert result is not None, "classify_grade should return a value"
            assert "sub" in result.lower(), \
                f"Grade {test_grade} should be Sub-economic (must contain 'sub')"
        finally:
            sys.path.pop(0)

    def test_classify_invalid_negative(self):
        """Should classify negative grades as Invalid"""
        sys.path.insert(0, str(SRC_DIR))
        try:
            from lab2_classify import classify_grade
            result = classify_grade(-1.0)
            assert result is not None, "classify_grade should return a value"
            assert "invalid" in result.lower(), "Negative grade should be Invalid"
        finally:
            sys.path.pop(0)

    def test_classify_zero_grade(self, grade_thresholds):
        """Should handle zero grade (valid, sub-economic)"""
        sys.path.insert(0, str(SRC_DIR))
        try:
            from lab2_classify import classify_grade
            result = classify_grade(0)
            assert result is not None, "classify_grade should return a value"
            # Zero is valid but sub-economic (not "Invalid")
            assert "invalid" not in result.lower(), "Zero grade should not be Invalid"
        finally:
            sys.path.pop(0)


# ============================================================================
# Task 2: Sample Processing Tests
# ============================================================================

class TestTask2Processing:
    """Tests for Task 2: Sample processing with for loops"""

    def test_processing_file_exists(self):
        """lab2_processing.py file should exist"""
        assert (SRC_DIR / "lab2_processing.py").exists(), "lab2_processing.py not found in src/"

    def test_processing_runs_without_error(self):
        """lab2_processing.py should run without errors"""
        result = run_script("lab2_processing.py")
        assert result.returncode == 0, f"Script failed with error: {result.stderr}"

    def test_process_samples_function_exists(self):
        """process_samples function should be defined"""
        sys.path.insert(0, str(SRC_DIR))
        try:
            from lab2_processing import process_samples
            assert callable(process_samples), "process_samples should be a function"
        finally:
            sys.path.pop(0)

    def test_process_samples_returns_dict(self, test_samples):
        """process_samples should return a dictionary"""
        sys.path.insert(0, str(SRC_DIR))
        try:
            from lab2_processing import process_samples
            result = process_samples(test_samples)
            assert result is not None, "process_samples should return a value"
            assert isinstance(result, dict), "process_samples should return a dict"
        finally:
            sys.path.pop(0)

    def test_process_samples_has_required_keys(self, test_samples):
        """process_samples should return dict with required keys"""
        sys.path.insert(0, str(SRC_DIR))
        try:
            from lab2_processing import process_samples
            result = process_samples(test_samples)
            required_keys = ['count', 'total', 'average', 'max_grade', 'min_grade']
            for key in required_keys:
                assert key in result, f"Result should contain '{key}'"
        finally:
            sys.path.pop(0)

    def test_process_samples_skips_negative(self):
        """process_samples should skip negative values"""
        sys.path.insert(0, str(SRC_DIR))
        try:
            from lab2_processing import process_samples
            samples = [1.0, -1.0, 2.0, -2.0, 3.0]
            result = process_samples(samples)
            # Should count only 3 valid samples (1.0, 2.0, 3.0)
            assert result['count'] == 3, "Should only count non-negative samples"
        finally:
            sys.path.pop(0)

    def test_count_by_category_function_exists(self):
        """count_by_category function should be defined"""
        sys.path.insert(0, str(SRC_DIR))
        try:
            from lab2_processing import count_by_category
            assert callable(count_by_category), "count_by_category should be a function"
        finally:
            sys.path.pop(0)

    def test_count_by_category_returns_dict(self, test_samples, grade_thresholds):
        """count_by_category should return a dictionary"""
        sys.path.insert(0, str(SRC_DIR))
        try:
            from lab2_processing import count_by_category
            result = count_by_category(test_samples, grade_thresholds)
            assert result is not None, "count_by_category should return a value"
            assert isinstance(result, dict), "count_by_category should return a dict"
        finally:
            sys.path.pop(0)

    def test_filter_samples_function_exists(self):
        """filter_samples function should be defined"""
        sys.path.insert(0, str(SRC_DIR))
        try:
            from lab2_processing import filter_samples
            assert callable(filter_samples), "filter_samples should be a function"
        finally:
            sys.path.pop(0)

    def test_filter_samples_returns_list(self):
        """filter_samples should return a list"""
        sys.path.insert(0, str(SRC_DIR))
        try:
            from lab2_processing import filter_samples
            result = filter_samples([1.0, 2.0, 3.0], min_grade=0)
            assert result is not None, "filter_samples should return a value"
            assert isinstance(result, list), "filter_samples should return a list"
        finally:
            sys.path.pop(0)

    def test_filter_samples_filters_by_min(self):
        """filter_samples should exclude samples below min_grade"""
        sys.path.insert(0, str(SRC_DIR))
        try:
            from lab2_processing import filter_samples
            samples = [0.5, 1.0, 1.5, 2.0, 2.5]
            result = filter_samples(samples, min_grade=1.5)
            if result is not None:
                assert 0.5 not in result, "Should exclude 0.5 (below min_grade 1.5)"
                assert 1.0 not in result, "Should exclude 1.0 (below min_grade 1.5)"
                assert 1.5 in result, "Should include 1.5 (at min_grade)"
                assert 2.0 in result, "Should include 2.0 (above min_grade)"
        finally:
            sys.path.pop(0)

    def test_filter_samples_filters_by_max(self):
        """filter_samples should exclude samples above max_grade"""
        sys.path.insert(0, str(SRC_DIR))
        try:
            from lab2_processing import filter_samples
            samples = [0.5, 1.0, 1.5, 2.0, 2.5]
            result = filter_samples(samples, min_grade=0, max_grade=2.0)
            if result is not None:
                assert 2.5 not in result, "Should exclude 2.5 (above max_grade 2.0)"
                assert 0.5 in result, "Should include 0.5 (within range)"
        finally:
            sys.path.pop(0)


# ============================================================================
# Task 3: Input Validation Tests
# ============================================================================

class TestTask3Validation:
    """Tests for Task 3: Input validation with while loops"""

    def test_validation_file_exists(self):
        """lab2_validation.py file should exist"""
        assert (SRC_DIR / "lab2_validation.py").exists(), "lab2_validation.py not found in src/"

    def test_get_valid_grade_function_exists(self):
        """get_valid_grade function should be defined"""
        sys.path.insert(0, str(SRC_DIR))
        try:
            from lab2_validation import get_valid_grade
            assert callable(get_valid_grade), "get_valid_grade should be a function"
        finally:
            sys.path.pop(0)

    def test_get_valid_grade_accepts_valid(self):
        """get_valid_grade should accept a valid grade on first try"""
        result = run_script("lab2_validation.py", input_data="50\n250\nmedium\n25\n50\n75\n")
        # If the function works, it should not print "Invalid" for the first valid input
        assert result.returncode == 0, f"Script failed: {result.stderr}"

    def test_get_valid_grade_accepts_valid_input(self, monkeypatch):
        """get_valid_grade should return a valid float when given valid input"""
        sys.path.insert(0, str(SRC_DIR))
        try:
            import importlib
            import lab2_validation
            importlib.reload(lab2_validation)
            monkeypatch.setattr('builtins.input', lambda *args: '75.5')
            result = lab2_validation.get_valid_grade()
            if result is not None:
                assert isinstance(result, (int, float)), "get_valid_grade should return a number"
                assert result == 75.5, "get_valid_grade should return the entered value"
        finally:
            sys.path.pop(0)

    def test_get_valid_depth_function_exists(self):
        """get_valid_depth function should be defined"""
        sys.path.insert(0, str(SRC_DIR))
        try:
            from lab2_validation import get_valid_depth
            assert callable(get_valid_depth), "get_valid_depth should be a function"
        finally:
            sys.path.pop(0)

    def test_get_valid_depth_accepts_valid_input(self, monkeypatch):
        """get_valid_depth should return a valid int when given valid input"""
        sys.path.insert(0, str(SRC_DIR))
        try:
            import importlib
            import lab2_validation
            importlib.reload(lab2_validation)
            monkeypatch.setattr('builtins.input', lambda *args: '250')
            result = lab2_validation.get_valid_depth()
            if result is not None:
                assert isinstance(result, (int, float)), "get_valid_depth should return a number"
                assert result == 250, "get_valid_depth should return the entered value"
        finally:
            sys.path.pop(0)

    def test_get_valid_choice_function_exists(self):
        """get_valid_choice function should be defined"""
        sys.path.insert(0, str(SRC_DIR))
        try:
            from lab2_validation import get_valid_choice
            assert callable(get_valid_choice), "get_valid_choice should be a function"
        finally:
            sys.path.pop(0)

    def test_collect_samples_function_exists(self):
        """collect_samples function should be defined"""
        sys.path.insert(0, str(SRC_DIR))
        try:
            from lab2_validation import collect_samples
            assert callable(collect_samples), "collect_samples should be a function"
        finally:
            sys.path.pop(0)


# ============================================================================
# Task 4: Drilling Cost Calculator Tests
# ============================================================================

class TestTask4Calculator:
    """Tests for Task 4: Drilling cost calculator"""

    def test_calculator_file_exists(self):
        """lab2_calculator.py file should exist"""
        assert (SRC_DIR / "lab2_calculator.py").exists(), "lab2_calculator.py not found in src/"

    def test_calculator_runs_without_error(self):
        """lab2_calculator.py should run without errors"""
        result = run_script("lab2_calculator.py")
        assert result.returncode == 0, f"Script failed with error: {result.stderr}"

    def test_calculate_drilling_cost_exists(self):
        """calculate_drilling_cost function should be defined"""
        sys.path.insert(0, str(SRC_DIR))
        try:
            from lab2_calculator import calculate_drilling_cost
            assert callable(calculate_drilling_cost), "calculate_drilling_cost should be a function"
        finally:
            sys.path.pop(0)

    def test_calculator_tier1_only(self, base_rate):
        """Should calculate correctly for tier 1 only (0-200m)"""
        sys.path.insert(0, str(SRC_DIR))
        try:
            from lab2_calculator import calculate_drilling_cost
            # 100 meters at base rate, medium hardness (no adjustment)
            cost = calculate_drilling_cost(100, base_rate, 'medium')
            expected = 100 * base_rate
            assert cost is not None, "Function should return a value"
            assert abs(cost - expected) < 0.01, f"Expected {expected}, got {cost}"
        finally:
            sys.path.pop(0)

    def test_calculator_tier2(self, base_rate):
        """Should apply 25% surcharge for 200-500m"""
        sys.path.insert(0, str(SRC_DIR))
        try:
            from lab2_calculator import calculate_drilling_cost
            # 300 meters: 200 at base, 100 at 1.25x
            cost = calculate_drilling_cost(300, base_rate, 'medium')
            expected = (200 * base_rate) + (100 * base_rate * 1.25)
            assert cost is not None, "Function should return a value"
            assert abs(cost - expected) < 0.01, f"Expected {expected}, got {cost}"
        finally:
            sys.path.pop(0)

    def test_calculator_tier3(self, base_rate):
        """Should apply 50% surcharge for over 500m"""
        sys.path.insert(0, str(SRC_DIR))
        try:
            from lab2_calculator import calculate_drilling_cost
            # 600 meters: 200 at base, 300 at 1.25x, 100 at 1.5x
            cost = calculate_drilling_cost(600, base_rate, 'medium')
            expected = (200 * base_rate) + (300 * base_rate * 1.25) + (100 * base_rate * 1.5)
            assert cost is not None, "Function should return a value"
            assert abs(cost - expected) < 0.01, f"Expected {expected}, got {cost}"
        finally:
            sys.path.pop(0)

    def test_calculator_soft_discount(self, base_rate):
        """Should apply 10% discount for soft rock"""
        sys.path.insert(0, str(SRC_DIR))
        try:
            from lab2_calculator import calculate_drilling_cost
            cost_medium = calculate_drilling_cost(100, base_rate, 'medium')
            cost_soft = calculate_drilling_cost(100, base_rate, 'soft')
            expected_soft = cost_medium * 0.9
            assert cost_soft is not None, "Function should return a value"
            assert abs(cost_soft - expected_soft) < 0.01, \
                f"Soft rock should have 10% discount. Expected {expected_soft}, got {cost_soft}"
        finally:
            sys.path.pop(0)

    def test_calculator_hard_surcharge(self, base_rate):
        """Should apply 20% surcharge for hard rock"""
        sys.path.insert(0, str(SRC_DIR))
        try:
            from lab2_calculator import calculate_drilling_cost
            cost_medium = calculate_drilling_cost(100, base_rate, 'medium')
            cost_hard = calculate_drilling_cost(100, base_rate, 'hard')
            expected_hard = cost_medium * 1.2
            assert cost_hard is not None, "Function should return a value"
            assert abs(cost_hard - expected_hard) < 0.01, \
                f"Hard rock should have 20% surcharge. Expected {expected_hard}, got {cost_hard}"
        finally:
            sys.path.pop(0)

    def test_calculator_invalid_depth_negative(self, base_rate):
        """Should return -1 for negative depth"""
        sys.path.insert(0, str(SRC_DIR))
        try:
            from lab2_calculator import calculate_drilling_cost
            cost = calculate_drilling_cost(-100, base_rate, 'medium')
            assert cost == -1, "Negative depth should return -1"
        finally:
            sys.path.pop(0)

    def test_calculator_invalid_depth_zero(self, base_rate):
        """Should return -1 for zero depth"""
        sys.path.insert(0, str(SRC_DIR))
        try:
            from lab2_calculator import calculate_drilling_cost
            cost = calculate_drilling_cost(0, base_rate, 'medium')
            assert cost == -1, "Zero depth should return -1"
        finally:
            sys.path.pop(0)

    def test_calculator_invalid_hardness(self, base_rate):
        """Should return -1 for invalid hardness"""
        sys.path.insert(0, str(SRC_DIR))
        try:
            from lab2_calculator import calculate_drilling_cost
            cost = calculate_drilling_cost(100, base_rate, 'very_hard')
            assert cost == -1, "Invalid hardness should return -1"
        finally:
            sys.path.pop(0)

    def test_cost_breakdown_returns_dict(self, base_rate):
        """calculate_cost_breakdown should return a dict"""
        sys.path.insert(0, str(SRC_DIR))
        try:
            from lab2_calculator import calculate_cost_breakdown
            result = calculate_cost_breakdown(300, base_rate)
            assert result is not None, "calculate_cost_breakdown should return a value for valid depth"
            assert isinstance(result, dict), "calculate_cost_breakdown should return a dict"
        finally:
            sys.path.pop(0)

    def test_cost_breakdown_has_required_keys(self, base_rate):
        """calculate_cost_breakdown should have tier and subtotal keys"""
        sys.path.insert(0, str(SRC_DIR))
        try:
            from lab2_calculator import calculate_cost_breakdown
            result = calculate_cost_breakdown(600, base_rate)
            if result is not None:
                required_keys = [
                    'tier1_meters', 'tier1_cost',
                    'tier2_meters', 'tier2_cost',
                    'tier3_meters', 'tier3_cost',
                    'subtotal'
                ]
                for key in required_keys:
                    assert key in result, f"Breakdown should contain '{key}'"
        finally:
            sys.path.pop(0)

    def test_cost_breakdown_correct_total(self, base_rate):
        """calculate_cost_breakdown subtotal should match calculate_drilling_cost for medium"""
        sys.path.insert(0, str(SRC_DIR))
        try:
            from lab2_calculator import calculate_cost_breakdown, calculate_drilling_cost
            breakdown = calculate_cost_breakdown(600, base_rate)
            cost = calculate_drilling_cost(600, base_rate, 'medium')
            if breakdown is not None and cost is not None and cost != -1:
                assert abs(breakdown['subtotal'] - cost) < 0.01, \
                    f"Breakdown subtotal ({breakdown['subtotal']}) should match drilling cost ({cost}) for medium hardness"
        finally:
            sys.path.pop(0)

    def test_format_cost_report_returns_string(self, base_rate):
        """format_cost_report should return a string"""
        sys.path.insert(0, str(SRC_DIR))
        try:
            from lab2_calculator import format_cost_report
            result = format_cost_report(350, 18750.0, 'medium', base_rate)
            assert result is not None, "format_cost_report should return a value"
            assert isinstance(result, str), "format_cost_report should return a string"
        finally:
            sys.path.pop(0)

    def test_format_cost_report_contains_values(self, base_rate):
        """format_cost_report should contain depth and cost values"""
        sys.path.insert(0, str(SRC_DIR))
        try:
            from lab2_calculator import format_cost_report
            result = format_cost_report(350, 18750.0, 'medium', base_rate)
            if result is not None:
                assert '350' in result, "Report should contain the depth value"
                assert '18750' in result or '18,750' in result, \
                    "Report should contain the cost value"
        finally:
            sys.path.pop(0)


# ============================================================================
# Task 5: Statistics Report Tests
# ============================================================================

class TestTask5Statistics:
    """Tests for Task 5: Grade statistics report"""

    def test_statistics_file_exists(self):
        """lab2_statistics.py file should exist"""
        assert (SRC_DIR / "lab2_statistics.py").exists(), "lab2_statistics.py not found in src/"

    def test_statistics_runs_without_error(self):
        """lab2_statistics.py should run without errors"""
        result = run_script("lab2_statistics.py")
        assert result.returncode == 0, f"Script failed with error: {result.stderr}"

    def test_generate_report_function_exists(self):
        """generate_report function should be defined"""
        sys.path.insert(0, str(SRC_DIR))
        try:
            from lab2_statistics import generate_report
            assert callable(generate_report), "generate_report should be a function"
        finally:
            sys.path.pop(0)

    def test_find_anomalies_function_exists(self):
        """find_anomalies function should be defined"""
        sys.path.insert(0, str(SRC_DIR))
        try:
            from lab2_statistics import find_anomalies
            assert callable(find_anomalies), "find_anomalies should be a function"
        finally:
            sys.path.pop(0)

    def test_find_anomalies_returns_list(self, test_samples):
        """find_anomalies should return a list"""
        sys.path.insert(0, str(SRC_DIR))
        try:
            from lab2_statistics import find_anomalies
            result = find_anomalies(test_samples)
            if result is not None:  # Only test if implemented
                assert isinstance(result, list), "find_anomalies should return a list"
        finally:
            sys.path.pop(0)

    def test_find_anomalies_detects_outliers(self):
        """find_anomalies should detect obvious outliers"""
        sys.path.insert(0, str(SRC_DIR))
        try:
            from lab2_statistics import find_anomalies
            # Samples with one obvious outlier
            samples = [1.0, 1.5, 1.2, 1.3, 1.4, 10.0]  # 10.0 is clearly an outlier
            result = find_anomalies(samples, threshold_multiplier=2.0)
            if result is not None:  # Only test if implemented
                assert 10.0 in result, "Should detect 10.0 as an anomaly"
        finally:
            sys.path.pop(0)

    def test_grade_distribution_returns_dict(self):
        """get_grade_distribution should return a dict"""
        sys.path.insert(0, str(SRC_DIR))
        try:
            from lab2_statistics import get_grade_distribution
            samples = [0.5, 1.5, 2.5, 3.5, 4.5]
            result = get_grade_distribution(samples)
            assert result is not None, "get_grade_distribution should return a value"
            assert isinstance(result, dict), "get_grade_distribution should return a dict"
        finally:
            sys.path.pop(0)

    def test_grade_distribution_correct_bins(self):
        """get_grade_distribution should have num_bins entries"""
        sys.path.insert(0, str(SRC_DIR))
        try:
            from lab2_statistics import get_grade_distribution
            samples = [0.5, 1.5, 2.5, 3.5, 4.5]
            result = get_grade_distribution(samples, num_bins=5)
            if result is not None:
                assert len(result) == 5, \
                    f"Distribution should have 5 bins, got {len(result)}"
        finally:
            sys.path.pop(0)

    def test_grade_distribution_total_count(self):
        """Sum of all bin counts should equal number of valid samples in range"""
        sys.path.insert(0, str(SRC_DIR))
        try:
            from lab2_statistics import get_grade_distribution
            samples = [0.5, 1.5, 2.5, 3.5, 4.5]
            result = get_grade_distribution(samples, num_bins=5, grade_range=(0, 5))
            if result is not None:
                total = sum(result.values())
                assert total == len(samples), \
                    f"Total count ({total}) should equal number of samples ({len(samples)})"
        finally:
            sys.path.pop(0)

    def test_print_histogram_returns_string(self):
        """print_histogram should return a string (or print output)"""
        sys.path.insert(0, str(SRC_DIR))
        try:
            from lab2_statistics import print_histogram
            distribution = {'0.0-1.0': 2, '1.0-2.0': 3, '2.0-3.0': 1}
            result = print_histogram(distribution)
            # print_histogram may return a string or None (just printing)
            # We check that it does not raise an error; if it returns a string, validate it
            if result is not None:
                assert isinstance(result, str), "If print_histogram returns a value, it should be a string"
        finally:
            sys.path.pop(0)

    def test_print_histogram_not_empty(self, capsys):
        """print_histogram should produce some output"""
        sys.path.insert(0, str(SRC_DIR))
        try:
            from lab2_statistics import print_histogram
            distribution = {'0.0-1.0': 2, '1.0-2.0': 5, '2.0-3.0': 3}
            result = print_histogram(distribution)
            captured = capsys.readouterr()
            # Either the return value or printed output should be non-empty
            has_output = (result is not None and len(str(result)) > 0) or len(captured.out) > 0
            assert has_output, "print_histogram should produce some output (returned or printed)"
        finally:
            sys.path.pop(0)
