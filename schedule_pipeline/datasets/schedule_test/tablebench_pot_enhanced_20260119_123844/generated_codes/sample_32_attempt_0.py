import pandas as pd

df = pd.read_csv('table.csv')
# Sum the '2011 (imf)' column
total_gdp = df['2011 (imf)'].astype(int).sum()
print(f"Final Answer: {total_gdp}")