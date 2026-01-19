import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where urban percentage is greater than 50
urban_above_50 = df[df['urban , %'] > 50]
# Find the first (earliest) year
first_year = urban_above_50['year (january)'].min()
print(f"Final Answer: {first_year}")