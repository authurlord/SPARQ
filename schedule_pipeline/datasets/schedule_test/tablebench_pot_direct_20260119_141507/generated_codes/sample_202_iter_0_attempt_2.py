import pandas as pd

df = pd.read_csv('table.csv')

# Convert the 'bp 2nd comp (˚c)' and 'bp 3rd comp (˚c)' to float
df['bp 2nd comp (˚c)'] = pd.to_numeric(df['bp 2nd comp (˚c)'], errors='coerce')
df['bp 3rd comp (˚c)'] = pd.to_numeric(df['bp 3rd comp (˚c)'], errors='coerce')
df['bp azeo (˚c)'] = pd.to_numeric(df['bp azeo (˚c)'], errors='coerce')

# Calculate differences
df['diff_2nd'] = abs(df['bp azeo (˚c)'] - df['bp 2nd comp (˚c)'])
df['diff_3rd'] = abs(df['bp azeo (˚c)'] - df['bp 3rd comp (˚c)'])

# Compute average differences
avg_diff_2nd = df['diff_2nd'].mean()
avg_diff_3rd = df['diff_3rd'].mean()

if avg_diff_2nd > avg_diff_3rd:
    final_answer = "2nd component"
else:
    final_answer = "3rd component"

print(f"Final Answer: {final_answer}")