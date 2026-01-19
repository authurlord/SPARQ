import pandas as pd

df = pd.read_csv('table.csv')
# Sum the '2011 (imf)' column for total GDP
total_gdp = df['2011 (imf)'].sum()
print(f"Final Answer: {total_gdp}")