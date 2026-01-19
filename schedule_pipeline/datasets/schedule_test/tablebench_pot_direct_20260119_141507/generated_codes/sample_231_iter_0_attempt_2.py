import pandas as pd

df = pd.read_csv('table.csv')
# Extract the relevant columns
population = df['metropolitan population (2006) millions']
gdp_per_capita = df['gdp (ppp) us per capita']

# Calculate the correlation between population and GDP per capita
correlation = population.corr(gdp_per_capita)

print(f"Final Answer: {correlation:.3f}")