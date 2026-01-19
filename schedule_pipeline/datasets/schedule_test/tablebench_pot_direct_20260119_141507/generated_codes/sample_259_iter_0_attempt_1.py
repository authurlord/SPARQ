# Extract Manchu populations for "north" and "liaoning"
manchu_north = 3002873
manchu_liaoning = 5336895

# Calculate required increase
increase_needed = manchu_liaoning - manchu_north

# Calculate percentage increase
percentage_increase = (increase_needed / manchu_north) * 100

print(f"Final Answer: {percentage_increase:.2f}")