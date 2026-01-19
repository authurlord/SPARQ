import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'live births per year' and 'deaths per year' to numeric
df['live births per year'] = pd.to_numeric(df['live births per year'].str.replace(',', ''), errors='coerce')
df['deaths per year'] = pd.to_numeric(df['deaths per year'].str.replace(',', ''), errors='coerce')

avg_live_births = df['live births per year'].mean()
avg_deaths = df['deaths per year'].mean()

if avg_live_births > avg_deaths:
    primary_driver = 'live births per year'
else:
    primary_driver = 'deaths per year'

print(f"Final Answer: {primary_driver}")