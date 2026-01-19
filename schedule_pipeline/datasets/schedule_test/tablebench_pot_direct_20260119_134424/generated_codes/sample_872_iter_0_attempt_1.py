import pandas as pd

df = pd.read_csv('table.csv')
# Convert boiling point columns to float
df['bp 2nd comp (˚c)'] = pd.to_numeric(df['bp 2nd comp (˚c)'])
df['bp 3rd comp (˚c)'] = pd.to_numeric(df['bp 3rd comp (˚c)'])
# Calculate the absolute difference in boiling points
df['diff'] = (df['bp 2nd comp (˚c)'] - df['bp 3rd comp (˚c)']).abs()
# Find the row with the smallest difference
min_diff_row = df.loc[df['diff'].idxmin()]
# Extract the components and the difference
pair = f"{min_diff_row['2nd component']} - {min_diff_row['3rd component']}"
difference = min_diff_row['diff']
print(f"Final Answer: {pair}, {difference:.1f}")