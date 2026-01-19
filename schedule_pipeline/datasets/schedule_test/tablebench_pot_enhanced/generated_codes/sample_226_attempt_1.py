import pandas as pd

df = pd.read_csv('table.csv')

# Clean and convert columns to numeric
for col in ['live births per year', 'deaths per year', 'natural change per year']:
    df[col] = df[col].str.replace(' ', '').astype(int)

# Calculate correlation between natural change and live births, and deaths
corr_births = df['natural change per year'].corr(df['live births per year'])
corr_deaths = df['natural change per year'].corr(df['deaths per year'])

# Determine which correlation is stronger
if abs(corr_births) > abs(corr_deaths):
    result = "live births per year"
else:
    result = "deaths per year"

print(f"Final Answer: {result}")