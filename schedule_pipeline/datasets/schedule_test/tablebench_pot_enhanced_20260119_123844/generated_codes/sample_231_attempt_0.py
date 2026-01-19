import pandas as pd

df = pd.read_csv('table.csv')
# Sort by metropolitan population to observe the trend
sorted_df = df.sort_values(by='metropolitan population (2006) millions', ascending=False)
# Display the sorted data showing GDP per capita vs population
print(sorted_df[['metropolitan population (2006) millions', 'gdp (ppp) us per capita']])