import pandas as pd

df = pd.read_csv('table.csv')

# Convert relevant columns to numeric, coercing errors to NaN
df['area (km 2 )'] = pd.to_numeric(df['area (km 2 )'], errors='coerce')
df['2002 population'] = pd.to_numeric(df['2002 population'], errors='coerce')

# Calculate expected pop density from population and area
df['expected_pop_density'] = df['2002 population'] / df['area (km 2 )']

# Check if expected pop density matches actual pop density (within small tolerance)
tolerance = 1e-2  # small tolerance for floating-point comparison
match_count = 0
for i in range(len(df)):
    actual = df.at[i, 'pop density (km 2 )']
    expected = df.at[i, 'expected_pop_density']
    if pd.isna(actual) or pd.isna(expected):
        continue
    if abs(actual - expected) < tolerance:
        match_count += 1

# If most entries match, the factors have significant influence
if match_count >= len(df) * 0.8:
    final_answer = "area (km^2) and 2002 population"
else:
    final_answer = "no clear impact"

print(f"Final Answer: {final_answer}")