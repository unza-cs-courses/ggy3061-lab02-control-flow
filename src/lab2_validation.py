"""
Lab 2 Task 3: Input Validation with While Loops
GGY3061 - Geoscience Data Analysis

Implement input validation functions using while loops.
"""

# Your Name: [YOUR NAME]
# Student ID: [YOUR ID]


def get_valid_grade():
    """
    Prompt user for a valid grade value using a while loop.

    Valid grades are between 0 and 100 (inclusive).
    Keep prompting until a valid value is entered.

    Returns:
        float: Valid grade value

    Example interaction:
        Enter grade (0-100): -5
        Invalid! Grade must be between 0 and 100.
        Enter grade (0-100): 150
        Invalid! Grade must be between 0 and 100.
        Enter grade (0-100): 75.5
        (returns 75.5)
    """
    # TODO: Use while True loop
    # TODO: Try to convert input to float
    # TODO: Check if value is in valid range
    # TODO: Break loop and return when valid
    # TODO: Handle ValueError for non-numeric input
    pass


def get_valid_depth():
    """
    Prompt user for a valid depth value using a while loop.

    Valid depths are positive integers.
    Keep prompting until a valid value is entered.

    Returns:
        int: Valid depth value

    Example interaction:
        Enter depth (positive integer): abc
        Invalid! Please enter a positive integer.
        Enter depth (positive integer): -10
        Invalid! Depth must be positive.
        Enter depth (positive integer): 250
        (returns 250)
    """
    # TODO: Use while True loop with try/except
    # TODO: Convert input to int
    # TODO: Check if value is positive
    # TODO: Handle ValueError for non-integer input
    pass


def get_valid_choice(options):
    """
    Prompt user to select from a list of valid options.

    Args:
        options: List of valid string choices

    Returns:
        str: The selected valid option

    Example:
        options = ['soft', 'medium', 'hard']
        Enter choice (soft/medium/hard): very hard
        Invalid! Choose from: soft, medium, hard
        Enter choice (soft/medium/hard): medium
        (returns 'medium')
    """
    # TODO: Use while loop to prompt until valid choice
    # TODO: Convert input to lowercase for comparison
    pass


def collect_samples(num_samples):
    """
    Collect a specified number of valid sample grades.

    Args:
        num_samples: Number of samples to collect

    Returns:
        list: List of valid grade values (each between 0 and 100)

    Note: Uses get_valid_grade() for each sample.
    """
    # TODO: Use for loop to collect num_samples values
    # TODO: Call get_valid_grade() for each sample
    pass


if __name__ == "__main__":
    print("=== Input Validation Demo ===\n")

    # Demonstrate single value validation
    print("--- Get Valid Grade ---")
    grade = get_valid_grade()
    print(f"Valid grade entered: {grade}\n")

    print("--- Get Valid Depth ---")
    depth = get_valid_depth()
    print(f"Valid depth entered: {depth}\n")

    print("--- Get Valid Choice ---")
    options = ['soft', 'medium', 'hard']
    choice = get_valid_choice(options)
    print(f"Valid choice: {choice}\n")

    # Collect multiple samples
    print("--- Collecting 3 Samples ---")
    samples = collect_samples(3)
    print(f"Collected samples: {samples}")
