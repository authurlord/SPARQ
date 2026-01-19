import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'Percent Increase (Decrease)' to numeric, coercing errors to NaN
df['Percent Increase (Decrease)'] = pd.to_numeric(df['Percent Increase (Decrease)'], errors='coerce')
# Filter rows where the percentage increase is greater than 5
high_increase_count = df[df['Percent Increase (Decrease)'] > 5].shape[0]
print(f"Final Answer: {high_increase_count}")