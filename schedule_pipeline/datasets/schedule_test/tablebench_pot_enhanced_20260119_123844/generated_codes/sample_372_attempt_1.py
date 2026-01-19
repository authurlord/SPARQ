import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'Percent Increase (Decrease)' to numeric, coercing errors to NaN
df['Percent Increase (Decrease)'] = pd.to_numeric(df['Percent Increase (Decrease)'], errors='coerce')
# Drop rows where the value is 'nm (1)' or NaN
df_filtered = df.dropna(subset=['Percent Increase (Decrease)'])
# Filter for values > 5
count = df_filtered[df_filtered['Percent Increase (Decrease)'] > 5].shape[0]
print(f"Final Answer: {count}")