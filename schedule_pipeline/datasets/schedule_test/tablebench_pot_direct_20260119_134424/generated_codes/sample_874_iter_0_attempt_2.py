# Extract male population data from the table
males_10_19 = 21
males_20_29 = 12
males_60_69 = 6
males_70_79 = 8
males_80_plus = 3

# Calculate total males in 10-29 age group
total_males_10_29 = males_10_19 + males_20_29

# Calculate total males in 60+ age group
total_males_60_plus = males_60_69 + males_70_79 + males_80_plus

# Find the difference
difference = total_males_10_29 - total_males_60_plus

print(f"Final Answer: {difference}")