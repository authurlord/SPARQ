import pandas as pd

df = pd.read_csv('table.csv')
# Convert boiling point columns to float
df['bp 2nd comp (˚c)'] = pd.to_numeric(df['bp 2nd comp (˚c)'])
df['bp 3rd comp (˚c)'] = pd.to_numeric(df['bp 3rd comp (˚c)'])
df['bp azeo (˚c)'] = pd.to_numeric(df['bp azeo (˚c)'])

# Calculate absolute differences
df['diff_2nd'] = abs(df['bp azeo (˚c)'] - df['bp 2nd comp (˚c)'])
df['diff_3rd'] = abs(df['bp azeo (˚c)'] - df['bp 3rd comp (˚c)'])

# Count how many times each component's difference is smaller
impact_2nd = (df['diff_2nd'] < df['diff_3rd']).sum()
impact_3rd = (df['diff_3rd'] < df['diff_2nd']).sum()

# Determine which has greater impact
if impact_2nd > impact_3rd:
    print("Final Answer: 2nd component")
else:
    print("Final Answer: 3rd component")