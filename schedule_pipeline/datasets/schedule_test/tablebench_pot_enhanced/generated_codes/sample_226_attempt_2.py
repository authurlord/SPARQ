import pandas as pd

df = pd.read_csv('table.csv')

# Clean and convert 'live births per year' and 'deaths per year' to integers
df['live births per year'] = df['live births per year'].str.replace(' ', '').astype(int)
df['deaths per year'] = df['deaths per year'].str.replace(' ', '').astype(int)

# Extract natural change per year (already given)
natural_change = df['natural change per year'].str.replace(' ', '').astype(int)

# Compute correlation between natural change and live births
corr_births = natural_change.corr(df['live births per year'])

# Compute correlation between natural change and deaths
corr_deaths = natural_change.corr(df['deaths per year'])

# Determine which correlation is stronger (in absolute terms)
if abs(corr_births) > abs(corr_deaths):
    driver = "live births per year"
else:
    driver = "deaths per year"

print(f"Final Answer: {driver}")