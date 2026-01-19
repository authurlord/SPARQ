import pandas as pd

df = pd.read_csv('table.csv')

# Identify outliers based on key metrics
# Filter for extreme values: very low population, negative growth, or very high % of pop
# Check for negative growth rate
negative_growth = df[df['average relative annual growth (%)'] < 0]
# Check for very high % of population
high_pop_percent = df[df['% of pop'] > 25]
# Check for very low population projection
low_population = df[df['july 1 , 2013 projection'] < 1000000]

# Countries with significant deviations
deviations = []
if not negative_growth.empty:
    deviations.append(negative_growth.iloc[0]['country (or dependent territory)'])
if not high_pop_percent.empty:
    deviations.append(high_pop_percent.iloc[0]['country (or dependent territory)'])
if not low_population.empty:
    deviations.append(low_population.iloc[0]['country (or dependent territory)'])

# Remove duplicates and ensure only distinct countries are listed
unique_deviation_countries = list(set(deviations))

print(f"Final Answer: {unique_deviation_countries[0]}, {unique_deviation_countries[1]}")