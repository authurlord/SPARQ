import pandas as pd

df = pd.read_csv('table.csv')

# Clean and convert columns to numeric
df['live births per year'] = df['live births per year'].str.replace(' ', '').astype(int)
df['deaths per year'] = df['deaths per year'].str.replace(' ', '').astype(int)
df['natural change per year'] = df['natural change per year'].str.replace(' ', '').astype(int)

# Calculate correlation between natural change and live births
corr_births = df['natural change per year'].corr(df['live births per year'])

# Calculate correlation between natural change and deaths
corr_deaths = df['natural change per year'].corr(df['deaths per year'])

# Determine which correlation is stronger (closer to 1 in absolute value)
if abs(corr_births) > abs(corr_deaths):
    primary_driver = "live births per year"
else:
    primary_driver = "deaths per year"

print(f"Final Answer: {primary_driver}")