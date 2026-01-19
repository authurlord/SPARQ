import pandas as pd

df = pd.read_csv('table.csv')

# Convert boiling point columns to float
df['bp 2nd comp (˚c)'] = pd.to_numeric(df['bp 2nd comp (˚c)'], errors='coerce')
df['bp 3rd comp (˚c)'] = pd.to_numeric(df['bp 3rd comp (˚c)'], errors='coerce')
df['bp azeo (˚c)'] = pd.to_numeric(df['bp azeo (˚c)'], errors='coerce')

# Calculate absolute differences
diff_2nd = abs(df['bp azeo (˚c)'] - df['bp 2nd comp (˚c)'])
diff_3rd = abs(df['bp azeo (˚c)'] - df['bp 3rd comp (˚c)'])

# Compute average differences
avg_diff_2nd = diff_2nd.mean()
avg_diff_3rd = diff_3rd.mean()

# Determine which has greater impact
if avg_diff_2nd < avg_diff_3rd:
    result = "2nd component"
else:
    result = "3rd component"

print(f"Final Answer: {result}")