# Extract population data from the table
pop_0_9 = 41
pop_10_19 = 45
pop_20_29 = 47
pop_30_39 = 27
pop_40_49 = 38
pop_50_59 = 31
pop_60_69 = 20
pop_70_79 = 24
pop_80_plus = 14

# Calculate population aged 0-10
pop_0_10 = pop_0_9 + pop_10_19

# Calculate population aged 60+
pop_60_plus = pop_60_69 + pop_70_79 + pop_80_plus

# Calculate population aged 11-59
pop_11_59 = pop_10_19 + pop_20_29 + pop_30_39 + pop_40_49 + pop_50_59

# Calculate dependency ratio
dependency_ratio = (pop_0_10 + pop_60_plus) / pop_11_59

# Print the result rounded to two decimal places
print(f"Final Answer: {dependency_ratio:.2f}")