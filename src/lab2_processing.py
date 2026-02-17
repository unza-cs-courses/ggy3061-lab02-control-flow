"""
Lab 2 Task 2: Sample Processing with For Loops
GGY3061 - Geoscience Data Analysis

Implement functions that process sample data using for loops.
"""

# Your Name: [YOUR NAME]
# Student ID: [YOUR ID]


def process_samples(samples):
    """
    Process a list of sample grades and return statistics.

    Args:
        samples: List of numeric grade values

    Returns:
        dict: Dictionary containing:
            - 'count': total number of valid samples (>= 0)
            - 'total': sum of all valid grades
            - 'average': average of valid grades (or 0 if no valid samples)
            - 'max_grade': highest valid grade (or None if no valid samples)
            - 'min_grade': lowest valid grade (or None if no valid samples)

    Note: Negative grade values should be skipped (they are invalid).
    """
    # TODO: Initialize variables for tracking statistics
    # TODO: Use a for loop to iterate through samples
    # TODO: Skip negative values
    # TODO: Track count, sum, min, and max of valid samples
    # TODO: Calculate average (handle case of no valid samples)
    # TODO: Return dictionary with all statistics
    pass


def count_by_category(samples, thresholds):
    """
    Count samples in each category based on thresholds.

    Args:
        samples: List of grade values
        thresholds: Dict with 'high', 'medium', 'low' threshold values

    Returns:
        dict: Count of samples in each category
              {'high': n, 'medium': n, 'low': n, 'subeconomic': n, 'invalid': n}

    Categories:
        - 'high': grade >= thresholds['high']
        - 'medium': thresholds['medium'] <= grade < thresholds['high']
        - 'low': thresholds['low'] <= grade < thresholds['medium']
        - 'subeconomic': 0 <= grade < thresholds['low']
        - 'invalid': grade < 0
    """
    # TODO: Initialize count dictionary
    # TODO: Use for loop to iterate and classify each sample
    # TODO: Return the counts
    pass


def filter_samples(samples, min_grade=0, max_grade=None):
    """
    Filter samples to include only those within a grade range.

    Args:
        samples: List of grade values
        min_grade: Minimum grade to include (default 0)
        max_grade: Maximum grade to include (default None = no limit)

    Returns:
        list: Filtered list of samples within the range
    """
    # TODO: Use for loop with conditionals to filter samples
    pass


if __name__ == "__main__":
    # Replace with YOUR test_samples from README.md
    samples = [3.5, 2.1, 1.5, 0.8, -0.5, 4.2, 0.0, 1.9]

    # Replace with YOUR thresholds from README.md
    thresholds = {'high': 3.0, 'medium': 2.0, 'low': 1.0}

    print("=== Sample Processing Demo ===\n")

    stats = process_samples(samples)
    print("Sample Statistics:")
    if stats:
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")

    print("\nCategory Counts:")
    counts = count_by_category(samples, thresholds)
    if counts:
        for category, count in counts.items():
            print(f"  {category}: {count}")

    print("\nFiltered Samples (grade >= 1.0):")
    filtered = filter_samples(samples, min_grade=1.0)
    print(f"  {filtered}")
