import random

def generate_random_numbers(n):
    total = 100
    min_ = 5
    max_ = 45
    # Start with a single number between 15 and 35
    numbers = [random.randint(15, 35)]

    # Generate n-1 random numbers such that the sum is 100
    for i in range(n-1):
        current_sum = sum(numbers)
        remaining = total - current_sum

        # Adjust the min and max values dynamically based on the current sum
        if remaining > 0:
            min_value = max(min_, remaining - (n-i-2)*35)
            max_value = min(max_, remaining - (n-i-2)*15)
        else:
            max_value = min(min_, -remaining - (n-i-2)*15)
            min_value = max(max_, -remaining - (n-i-2)*35)

        # Generate a random number between min_value and max_value
        if min_value <= max_value:
            new_number = random.randint(min_value, max_value)
            numbers.append(new_number)

    # Shuffle the list to ensure randomness
    random.shuffle(numbers)

    return numbers
