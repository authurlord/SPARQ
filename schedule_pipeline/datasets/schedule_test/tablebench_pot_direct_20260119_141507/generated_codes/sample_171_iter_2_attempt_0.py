import pandas as pd

df = pd.read_csv('table.csv')

# Convert area and population to numeric, handling any parsing issues
df['area (km 2 )'] = pd.to_numeric(df['area (km 2 )'], errors='coerce')
df['2002 population'] = pd.to_numeric(df['2002 population'], errors='coerce')

# Compute expected pop density from area and population
df['expected_pop_density'] = df['2002 population'] / df['area (km 2 )']

# Compare with actual pop density
# Check if the expected value is close to actual value (within 1% tolerance)
tolerance = 0.01
df['match'] = abs(df['expected_pop_density'] - df['pop density (km 2 )']) <= (tolerance * df['pop density (km 2 )'])

# Count how many rows have a match
match_count = df['match'].sum()

# If all or most rows match, the factors have significant influence
if match_count >= len(df) * 0.9:
    final_answer = "area (km^2) and 2002 population"
else:
    final_answer = "no clear impact"

print(f"Final Answer: {final_answer}")