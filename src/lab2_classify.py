"""
Lab 2 Task 1: Grade Classification
GGY3061 - Geoscience Data Analysis

Implement the classify_grade function using if/elif/else statements.
"""

# Your Name: [YOUR NAME]
# Student ID: [YOUR ID]


def classify_grade(grade):
    """
    Classify an ore grade value into categories.

    Args:
        grade: Numeric grade value (can be float or int)

    Returns:
        str: Classification category
            - "High Grade" if grade >= high threshold
            - "Medium Grade" if grade >= medium threshold
            - "Low Grade" if grade >= low threshold
            - "Sub-economic" if grade >= 0 but below low threshold
            - "Invalid" if grade < 0

    Note: Check README.md for YOUR specific threshold values.
    """
    # TODO: Implement the classification logic
    # 1. First check if grade is negative (Invalid)
    # 2. Then check from highest to lowest threshold
    # 3. Use the thresholds from YOUR README.md
    pass


if __name__ == "__main__":
    # Test your function with sample values
    # Replace these with YOUR test_samples from README.md
    test_grades = [3.5, 2.1, 1.5, 0.8, -0.5, 4.2, 0.0]

    print("Grade Classification Results:")
    print("-" * 40)
    for grade in test_grades:
        result = classify_grade(grade)
        print(f"Grade {grade:6.2f} -> {result}")
