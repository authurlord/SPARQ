import pandas as pd

df = pd.read_csv('table.csv')
# Convert boiling point columns to numeric
df['bp 2nd comp (˚c)'] = pd.to_numeric(df['bp 2nd comp (˚c)'])
df['bp 3rd comp (˚c)'] = pd.to_numeric(df['bp 3rd comp (˚c)'])
df['bp azeo (˚c)'] = pd.to_numeric(df['bp azeo (˚c)'])

# Calculate absolute differences
df['diff_2nd'] = (df['bp azeo (˚c)'] - df['bp 2nd comp (˚c)']).abs()
df['diff_3rd'] = (df['bp azeo (˚c)'] - df['bp 3rd comp (˚c)']).abs()

# Average differences
avg_diff_2nd = df['diff_2nd'].mean()
avg_diff_3rd = df['diff_3rd'].mean()

# Determine which has greater impact (smaller average difference)
if avg_diff_2nd < avg_diff_3rd:
    impact = '2nd component'
else:
    impact = '3rd component'

print(f"Final Answer: {impact}")