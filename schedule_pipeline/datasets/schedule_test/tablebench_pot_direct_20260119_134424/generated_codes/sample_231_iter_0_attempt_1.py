import pandas as pd

df = pd.read_csv('table.csv')
# Convert columns to numeric for analysis
df['metropolitan population (2006) millions'] = pd.to_numeric(df['metropolitan population (2006) millions'])
df['gdp (ppp) us per capita'] = pd.to_numeric(df['gdp (ppp) us per capita'])

# Calculate correlation between population and GDP per capita
correlation = df['metropolitan population (2006) millions'].corr(df['gdp (ppp) us per capita'])

print(f"Final Answer: {correlation:.2f}")