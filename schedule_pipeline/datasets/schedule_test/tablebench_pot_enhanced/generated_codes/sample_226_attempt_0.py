import pandas as pd

df = pd.read_csv('table.csv')

# Clean and convert numerical columns
df['live births per year'] = df['live births per year'].str.replace(' ', '').astype(int)
df['deaths per year'] = df['deaths per year'].str.replace(' ', '').astype(int)
df['natural change per year'] = df['natural change per year'].str.replace(' ', '').astype(int)

# Calculate correlations
correlation_births = df['natural change per year'].corr(df['live births per year'])
correlation_deaths = df['natural change per year'].corr(df['deaths per year'])

# Determine which is more closely related
if abs(correlation_births) > abs(correlation_deaths):
    result = "live births per year"
else:
    result = "deaths per year"

print(f"Final Answer: {result}")