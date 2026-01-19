import pandas as pd

df = pd.read_csv('table.csv')
# Convert boiling point columns to float
df['bp 2nd comp (˚c)'] = pd.to_numeric(df['bp 2nd comp (˚c)'])
df['bp 3rd comp (˚c)'] = pd.to_numeric(df['bp 3rd comp (˚c)'])

# Calculate the absolute difference in boiling points
df['diff'] = abs(df['bp 2nd comp (˚c)'] - df['bp 3rd comp (˚c)'])

# Find the row with the smallest difference
min_diff_row = df.loc[df['diff'].idxmin()]

# Extract the components and the difference
component_2nd = min_diff_row['2nd component']
component_3rd = min_diff_row['3rd component']
difference = min_diff_row['diff']

print(f"Final Answer: {component_2nd} and {component_3rd}, {difference:.1f}")