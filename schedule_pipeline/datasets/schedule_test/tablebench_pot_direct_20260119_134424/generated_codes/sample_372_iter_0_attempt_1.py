import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'Percent Increase (Decrease)' to numeric, coercing errors to NaN
percent_increase = pd.to_numeric(df['Percent Increase (Decrease)'], errors='coerce')
# Count values greater than 5
count_greater_than_5 = (percent_increase > 5).sum()
print(f"Final Answer: {count_greater_than_5}")