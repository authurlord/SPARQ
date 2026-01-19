import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'Percent Increase (Decrease)' to numeric, coercing errors to NaN
df['Percent Increase (Decrease)'] = pd.to_numeric(df['Percent Increase (Decrease)'], errors='coerce')
# Drop rows with 'nm' or invalid entries
df_clean = df.dropna(subset=['Percent Increase (Decrease)'])
# Filter for percentage increase > 5
count_high_increase = df_clean[df_clean['Percent Increase (Decrease)'] > 5].shape[0]
print(f"Final Answer: {count_high_increase}")