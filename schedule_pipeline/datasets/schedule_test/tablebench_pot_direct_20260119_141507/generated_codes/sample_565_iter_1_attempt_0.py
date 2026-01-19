import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv('table.csv')

# Extract the relevant years between 1850 and 1870
relevant_years = [1850, 1855, 1860, 1865, 1870]

# Filter rows where Year is in relevant_years
filtered_df = df[df['Year'].isin(relevant_years)]

# Convert the second column (which has values like "8,385") to integers
# Remove commas and convert
filtered_df['value'] = filtered_df.iloc[:, 1].str.replace(',', '').astype(int)

# Sort by Year
filtered_df = filtered_df.sort_values(by='Year')

# Compute the differences between consecutive values
differences = filtered_df['value'].diff()

# Find the index of the maximum difference (which corresponds to the year before the jump)
max_diff_idx = differences.idxmax()  # This gives the index in the original filtered_df

# The year with the highest increase is the year corresponding to the start of the jump
# i.e., the year at the index before the jump
# But actually, the difference is between year_i and year_{i+1}
# So the year with the highest increase is the year at index i (the start of the jump)

# Get the value at that year
year_with_max_increase = filtered_df.loc[differences.idxmax(), 'Year']
value_at_that_year = filtered_df.loc[filtered_df['Year'] == year_with_max_increase, 'value'].iloc[0]

print(f"Final Answer: {value_at_that_year}")