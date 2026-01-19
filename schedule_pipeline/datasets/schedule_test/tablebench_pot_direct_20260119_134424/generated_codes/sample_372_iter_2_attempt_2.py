import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'Percent Increase (Decrease)' to numeric, coercing errors to NaN
df['Percent Increase (Decrease)'] = pd.to_numeric(df['Percent Increase (Decrease)'], errors='coerce')
# Filter rows where percentage increase > 5
high_increase = df[df['Percent Increase (Decrease)'] > 5]
# Count the number of such rows
count = len(high_increase)
print(f"Final Answer: {count}")