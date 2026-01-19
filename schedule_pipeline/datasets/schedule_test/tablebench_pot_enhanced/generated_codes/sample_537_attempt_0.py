import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'urban , %' to numeric and filter for values > 50
urban_above_50 = df[df['urban , %'].astype(int) > 50]
# Find the first year this occurred
first_year = urban_above_50['year (january)'].min()
print(f"Final Answer: {first_year}")