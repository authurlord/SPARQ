import pandas as pd

df = pd.read_csv('table.csv')
# Sort by metropolitan population to analyze the trend
sorted_df = df.sort_values(by='metropolitan population (2006) millions')
print(sorted_df[['metropolitan population (2006) millions', 'gdp (ppp) us per capita']])