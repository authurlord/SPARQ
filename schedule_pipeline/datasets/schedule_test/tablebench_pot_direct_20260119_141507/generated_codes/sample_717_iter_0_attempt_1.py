import pandas as pd

df = pd.read_csv('table.csv')
# Find the economy with the highest GDP per capita as of 2012 (PPP)
max_gdp_per_capita = df.loc[df['gdp per capita as of 2012 (ppp)'].idxmax(), 'economy']
print(f"Final Answer: {max_gdp_per_capita}")