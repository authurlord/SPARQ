import pandas as pd

df = pd.read_csv('table.csv')

# Convert boiling point columns to float
df['bp 2nd comp (˚c)'] = pd.to_numeric(df['bp 2nd comp (˚c)'], errors='coerce')
df['bp 3rd comp (˚c)'] = pd.to_numeric(df['bp 3rd comp (˚c)'], errors='coerce')

# Calculate the difference in boiling points
df['bp_diff'] = abs(df['bp 2nd comp (˚c)'] - df['bp 3rd comp (˚c)'])

# Find the row with minimum difference
min_diff_row = df.loc[df['bp_diff'].idxmin()]
pair = (min_diff_row['2nd component'], min_diff_row['3rd component'])
difference = min_diff_row['bp_diff']

print(f"Final Answer: {pair[0]}, {pair[1]}, {difference:.1f}")