import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'Percent Increase (Decrease)' to numeric, coercing errors to NaN
df['Percent Increase (Decrease) (In cents, except percentage changes)'] = pd.to_numeric(df['Percent Increase (Decrease) (In cents, except percentage changes)'], errors='coerce')
# Filter rows where percentage increase > 5
high_increase = df[df['Percent Increase (Decrease) (In cents, except percentage changes)'] > 5]
count = len(high_increase)
print(f"Final Answer: {count}")