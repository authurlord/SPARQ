import pandas as pd

df = pd.read_csv('table.csv')

# Convert boiling point columns to float
df['bp 2nd comp (˚c)'] = pd.to_numeric(df['bp 2nd comp (˚c)'], errors='coerce')
df['bp 3rd comp (˚c)'] = pd.to_numeric(df['bp 3rd comp (˚c)'], errors='coerce')

# Calculate the difference in boiling points
df['bp_diff'] = df['bp 3rd comp (˚c)'] - df['bp 2nd comp (˚c)']

# Find the row with the smallest difference
min_diff_row = df.loc[df['bp_diff'].idxmin()]

# Extract the 2nd and 3rd components and the difference
second_component = min_diff_row['2nd component']
third_component = min_diff_row['3rd component']
difference = min_diff_row['bp_diff']

print(f"Final Answer: {second_component}, {third_component}, {difference:.1f}")