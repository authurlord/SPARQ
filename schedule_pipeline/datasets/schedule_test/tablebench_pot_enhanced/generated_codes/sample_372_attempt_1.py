import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'Percent Increase (Decrease)' to numeric, coercing errors (like 'nm') to NaN
df['Percent Increase (Decrease)'] = pd.to_numeric(df['Percent Increase (Decrease)'], errors='coerce')
# Filter for values > 5 and count
count = df[df['Percent Increase (Decrease)'] > 5].shape[0]
print(f"Final Answer: {count}")