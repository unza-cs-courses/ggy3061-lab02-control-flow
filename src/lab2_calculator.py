"""
Lab 2 Task 4: Drilling Cost Calculator
GGY3061 - Geoscience Data Analysis

Implement a drilling cost calculator function with parameters and conditional logic.
"""

# Your Name: [YOUR NAME]
# Student ID: [YOUR ID]


def calculate_drilling_cost(depth, base_rate=50, hardness='medium'):
    """
    Calculate the total drilling cost based on depth and conditions.

    Args:
        depth: Drilling depth in meters (positive integer)
        base_rate: Base cost per meter (default: 50, use YOUR value from README.md)
        hardness: Rock hardness - 'soft', 'medium', or 'hard'

    Returns:
        float: Total drilling cost, or -1 if invalid input

    Pricing Rules:
    - First 200 meters: base_rate per meter
    - 200-500 meters: base_rate * 1.25 per meter (25% surcharge)
    - Over 500 meters: base_rate * 1.5 per meter (50% surcharge)

    Example for 600m at $50/m base rate:
        First 200m: 200 * 50 = $10,000
        Next 300m (200-500): 300 * 50 * 1.25 = $18,750
        Last 100m (500-600): 100 * 50 * 1.5 = $7,500
        Subtotal: $36,250

    Hardness Adjustments (applied to subtotal):
    - 'soft': 10% discount (multiply by 0.9)
    - 'medium': no adjustment (multiply by 1.0)
    - 'hard': 20% surcharge (multiply by 1.2)

    Invalid inputs (return -1):
    - Negative or zero depth
    - Non-positive base_rate
    - Invalid hardness value (not 'soft', 'medium', or 'hard')

    Note: Check README.md for YOUR base_rate value.
    """
    # TODO: Validate inputs (return -1 for invalid)
    # TODO: Calculate cost for each depth tier
    # TODO: Apply hardness adjustment
    # TODO: Return total cost
    pass


def calculate_cost_breakdown(depth, base_rate=50):
    """
    Calculate a detailed cost breakdown by depth tier.

    Args:
        depth: Drilling depth in meters
        base_rate: Base cost per meter

    Returns:
        dict: Breakdown with keys:
            - 'tier1_meters': meters in first tier (0-200)
            - 'tier1_cost': cost for first tier
            - 'tier2_meters': meters in second tier (200-500)
            - 'tier2_cost': cost for second tier
            - 'tier3_meters': meters in third tier (500+)
            - 'tier3_cost': cost for third tier
            - 'subtotal': sum of all tier costs

    Returns None if depth <= 0.
    """
    # TODO: Calculate meters and cost in each tier
    pass


def format_cost_report(depth, cost, hardness, base_rate):
    """
    Format a drilling cost as a report string.

    Args:
        depth: Drilling depth
        cost: Calculated cost
        hardness: Rock hardness
        base_rate: Base rate used

    Returns:
        str: Formatted multi-line report

    Example output:
        ================================
        DRILLING COST ESTIMATE
        ================================
        Depth: 350 meters
        Base Rate: $50/meter
        Rock Hardness: medium
        --------------------------------
        Estimated Cost: $18,750.00
        ================================
    """
    # TODO: Create a formatted string report
    pass


if __name__ == "__main__":
    print("=== Drilling Cost Calculator ===\n")

    # Replace with YOUR base_rate from README.md
    base_rate = 50

    # Replace with YOUR drilling_depths from README.md
    test_depths = [150, 350, 600, 800]
    hardness_options = ['soft', 'medium', 'hard']

    print(f"Base Rate: ${base_rate}/meter\n")

    for depth in test_depths:
        print(f"Depth: {depth}m")
        for hardness in hardness_options:
            cost = calculate_drilling_cost(depth, base_rate, hardness)
            if cost is None:
                print(f"  {hardness:8s}: Not yet implemented")
            elif cost >= 0:
                print(f"  {hardness:8s}: ${cost:,.2f}")
            else:
                print(f"  {hardness:8s}: Invalid input")
        print()

    # Test invalid inputs
    print("=== Invalid Input Tests ===")
    print(f"Negative depth (-100): {calculate_drilling_cost(-100, base_rate)}")
    print(f"Zero depth (0): {calculate_drilling_cost(0, base_rate)}")
    print(f"Invalid hardness: {calculate_drilling_cost(200, base_rate, 'very_hard')}")
