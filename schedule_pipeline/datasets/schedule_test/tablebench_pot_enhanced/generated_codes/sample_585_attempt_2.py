import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'number of tropical storms' to numeric
df['number of tropical storms'] = pd.to_numeric(df['number of tropical storms'])

# Calculate the difference in tropical storms from the previous year
df['increase'] = df['number of tropical storms'].diff()

# Find the row with the maximum increase (excluding the first year)
max_increase_year = df.loc[df['increase'].idxmax()]

print(f"Final Answer: {max_increase_year['year']}")