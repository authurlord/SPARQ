# Extract population data from the table
pop_0_to_9 = 41
pop_10_to_19 = 45
pop_20_to_29 = 47
pop_30_to_39 = 27
pop_40_to_49 = 38
pop_50_to_59 = 31
pop_60_to_69 = 20
pop_70_to_79 = 24
pop_80_plus = 14

# Calculate dependency ratio
pop_0_to_10 = pop_0_to_9 + pop_10_to_19
pop_60_plus = pop_60_to_69 + pop_70_to_79 + pop_80_plus
pop_11_to_59 = pop_10_to_19 + pop_20_to_29 + pop_30_to_39 + pop_40_to_49 + pop_50_to_59

dependency_ratio = (pop_0_to_10 + pop_60_plus) / pop_11_to_59

print(f"Final Answer: {dependency_ratio:.2f}")