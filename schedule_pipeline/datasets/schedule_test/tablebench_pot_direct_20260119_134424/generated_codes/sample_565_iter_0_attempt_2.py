import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Clean and convert the values (remove commas and convert to int)
df['Year_1'] = df['Year_1'].str.replace(',', '').astype(int)

# Filter data for years 1850 to 1870
df_filtered = df[(df['Year'] >= '1850') & (df['Year'] <= '1870')]

# Sort by Year to ensure chronological order
df_filtered = df_filtered.sort_values(by='Year')

# Calculate the increase (difference) between consecutive years
df_filtered['increase'] = df_filtered['Year_1'].diff()

# Find the row with the maximum increase
max_increase_row = df_filtered.loc[df_filtered['increase'].idxmax()]

# Get the value (Year_1) for the year with the highest increase
value_with_highest_increase = max_increase_row['Year_1']

print(f"Final Answer: {value_with_highest_increase}")