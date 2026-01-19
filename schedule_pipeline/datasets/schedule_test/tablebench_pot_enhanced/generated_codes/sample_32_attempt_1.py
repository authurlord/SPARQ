import pandas as pd

df = pd.read_csv('table.csv')
# Sum the '2011 (imf)' column to get total GDP in thousands of USD
total_gdp = df['2011 (imf)'].astype(int).sum()
print(f"Final Answer: {total_gdp}")