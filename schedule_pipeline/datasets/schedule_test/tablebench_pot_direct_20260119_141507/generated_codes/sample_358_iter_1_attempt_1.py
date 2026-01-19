import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where 'indians admitted' is greater than 25000
years_above_25000 = df[df['indians admitted'] > 25000]
# Count the number of such years
count_years = len(years_above_25000)
print(f"Final Answer: {count_years}")