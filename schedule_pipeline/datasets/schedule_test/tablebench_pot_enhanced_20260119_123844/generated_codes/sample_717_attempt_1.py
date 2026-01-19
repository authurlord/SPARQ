import pandas as pd

df = pd.read_csv('table.csv')
# Find the economy with the highest GDP per capita in 2012 (PPP)
max_gdp_per_capita = df.loc[df['gdp per capita as of 2012 (ppp)'].idxmax()]
economy = max_gdp_per_capita['economy']
print(f"Final Answer: {economy}")