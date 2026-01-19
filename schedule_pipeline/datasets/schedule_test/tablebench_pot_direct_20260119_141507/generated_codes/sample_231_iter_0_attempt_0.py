import pandas as pd

df = pd.read_csv('table.csv')
# Calculate the correlation between metropolitan population (millions) and GDP (PPP) per capita
correlation = df['metropolitan population (2006) millions'].corr(df['gdp (ppp) us per capita'])
print(f"Final Answer: {correlation:.3f}")