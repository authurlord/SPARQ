import pandas as pd

df = pd.read_csv('table.csv')

# Filter rows where Year_1 is between 1850 and 1870 (inclusive)
df_filtered = df[(df['Year_1'] >= '1850') & (df['Year_1'] <= '1870')]

# Convert the values in '-_1' to integers (remove commas)
df_filtered['-_1'] = df_filtered['-_1'].str.replace(',', '').astype(int)

# Sort by year to ensure chronological order
df_filtered = df_filtered.sort_values(by='Year_1')

# Calculate year-on-year increase
df_filtered['increase'] = df_filtered['-_1'].diff()

# Find the row with the maximum increase
max_increase_row = df_filtered.loc[df_filtered['increase'].idxmax()]

# Get the value corresponding to the year with the highest increase
value_with_highest_increase = max_increase_row['-_1']

print(f"Final Answer: {value_with_highest_increase}")