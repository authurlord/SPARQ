import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'number of tropical storms' to integer
df['number of tropical storms'] = pd.to_numeric(df['number of tropical storms'])
# Calculate the year-on-year increase
df['increase'] = df['number of tropical storms'].diff()
# Find the row with the maximum increase (excluding the first year)
max_increase_row = df.loc[df['increase'].idxmax()]
# Extract the year with the maximum increase
year_with_max_increase = max_increase_row['year']
print(f"Final Answer: {year_with_max_increase}")