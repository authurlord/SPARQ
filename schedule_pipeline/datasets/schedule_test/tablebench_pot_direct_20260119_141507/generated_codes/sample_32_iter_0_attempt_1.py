import pandas as pd

df = pd.read_csv('table.csv')
# Sum the '2011 (imf)' column to get total GDP
total_gdp_2011 = df['2011 (imf)'].sum()
print(f"Final Answer: {total_gdp_2011}")