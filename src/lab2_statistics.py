"""
Lab 2 Task 5: Grade Statistics Report
GGY3061 - Geoscience Data Analysis

Combine control structures and functions to generate comprehensive reports.
"""

# Your Name: [YOUR NAME]
# Student ID: [YOUR ID]

# Import functions from other modules
# Note: These imports will work when the other files are completed
try:
    from lab2_classify import classify_grade
    from lab2_processing import process_samples, count_by_category
except ImportError:
    # Placeholder functions if imports fail
    def classify_grade(grade):
        return "Unknown"

    def process_samples(samples):
        return {'count': 0, 'total': 0, 'average': 0, 'max_grade': None, 'min_grade': None}

    def count_by_category(samples, thresholds):
        return {'high': 0, 'medium': 0, 'low': 0, 'subeconomic': 0, 'invalid': 0}


def generate_report(samples, thresholds):
    """
    Generate a comprehensive grade statistics report.

    Args:
        samples: List of grade values
        thresholds: Dict with 'high', 'medium', 'low' threshold values

    Returns:
        str: Formatted multi-line report containing:
            - Header with title
            - Summary statistics (count, average, min, max)
            - Category breakdown with counts and percentages
            - List of high-grade samples

    Example output:
        ================================================
        GRADE STATISTICS REPORT
        ================================================

        SUMMARY STATISTICS
        ------------------
        Total Samples: 10
        Valid Samples: 8
        Average Grade: 2.35
        Minimum Grade: 0.50
        Maximum Grade: 4.20

        CATEGORY BREAKDOWN
        ------------------
        High Grade:    2 (25.0%)
        Medium Grade:  3 (37.5%)
        Low Grade:     2 (25.0%)
        Sub-economic:  1 (12.5%)
        Invalid:       2

        HIGH-GRADE SAMPLES
        ------------------
        - Grade 4.20
        - Grade 3.50

        ================================================
    """
    # TODO: Use process_samples to get statistics
    # TODO: Use count_by_category for breakdown
    # TODO: Build formatted string report
    # TODO: Include list of high-grade samples using for loop
    pass


def find_anomalies(samples, threshold_multiplier=2.0):
    """
    Find grade values that are anomalously high.

    An anomaly is defined as a valid grade > (average * threshold_multiplier)
    Only valid grades (>= 0) are considered.

    Args:
        samples: List of grade values
        threshold_multiplier: Multiplier for anomaly detection (default 2.0)

    Returns:
        list: List of anomalous grade values, sorted in descending order

    Example:
        samples = [1.0, 2.0, 1.5, 8.0, 1.2]
        Average of valid samples = 2.74
        Anomaly threshold = 2.74 * 2.0 = 5.48
        Returns: [8.0] (only 8.0 exceeds 5.48)
    """
    # TODO: Filter out invalid samples (negative values)
    # TODO: Calculate average of valid samples
    # TODO: Calculate anomaly threshold
    # TODO: Find and return samples exceeding threshold
    # TODO: Sort in descending order
    pass


def get_grade_distribution(samples, num_bins=5, grade_range=(0, 5)):
    """
    Create a simple distribution of grades across bins.

    Args:
        samples: List of grade values
        num_bins: Number of bins to divide range into (default 5)
        grade_range: Tuple of (min, max) grade range (default 0-5)

    Returns:
        dict: Dictionary with bin labels as keys and counts as values

    Example with num_bins=5, range=(0,5):
        Bins: 0-1, 1-2, 2-3, 3-4, 4-5
        Returns: {'0.0-1.0': 3, '1.0-2.0': 5, '2.0-3.0': 2, ...}
    """
    # TODO: Calculate bin width
    # TODO: Create bins and count samples in each
    pass


def print_histogram(distribution, max_width=40):
    """
    Print a simple text-based histogram.

    Args:
        distribution: Dict from get_grade_distribution
        max_width: Maximum width of histogram bars

    Example output:
        Grade Distribution:
        0.0-1.0 | ████████ (3)
        1.0-2.0 | ████████████████████ (8)
        2.0-3.0 | ████████████ (5)
        3.0-4.0 | ████ (2)
        4.0-5.0 | ██ (1)
    """
    # TODO: Find maximum count for scaling
    # TODO: Print each bin with scaled bar
    pass


if __name__ == "__main__":
    # Replace with YOUR test_samples from README.md
    samples = [3.5, 2.1, 1.5, 0.8, -0.5, 4.2, 0.0, 1.9, 2.8, 0.5]

    # Replace with YOUR thresholds from README.md
    thresholds = {'high': 3.0, 'medium': 2.0, 'low': 1.0}

    # Generate and print report
    report = generate_report(samples, thresholds)
    if report:
        print(report)
    else:
        print("Report generation not yet implemented")

    # Find anomalies
    print("\n=== Anomaly Detection ===")
    anomalies = find_anomalies(samples)
    if anomalies is not None:
        if anomalies:
            print(f"Found {len(anomalies)} anomalous value(s):")
            for value in anomalies:
                print(f"  - {value}")
        else:
            print("No anomalies detected.")
    else:
        print("Anomaly detection not yet implemented")

    # Show distribution
    print("\n=== Grade Distribution ===")
    distribution = get_grade_distribution(samples)
    if distribution:
        print_histogram(distribution)
    else:
        print("Distribution not yet implemented")
