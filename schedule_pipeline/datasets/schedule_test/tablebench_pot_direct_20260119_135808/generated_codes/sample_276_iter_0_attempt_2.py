import pandas as pd

df = pd.read_csv('table.csv')
# Extract GDP per capita values
gdp_per_capita_west_germany = float(df[df['member countries'] == 'west germany']['gdp per capita (us)'].values[0])
gdp_per_capita_france = float(df[df['member countries'] == 'france']['gdp per capita (us)'].values[0])

# Calculate minimum increase required
min_increase = gdp_per_capita_france - gdp_per_capita_west_germany + 1  # +1 to surpass, not just equal

print(f"Final Answer: {min_increase:.0f}")