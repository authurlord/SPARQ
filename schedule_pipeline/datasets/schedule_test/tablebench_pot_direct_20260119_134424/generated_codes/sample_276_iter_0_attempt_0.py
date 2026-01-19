import pandas as pd

df = pd.read_csv('table.csv')
# Extract GDP per capita for West Germany and France
gdp_per_capita_west_germany = float(df[df['member countries'] == 'west germany']['gdp per capita (us)'].values[0])
gdp_per_capita_france = float(df[df['member countries'] == 'france']['gdp per capita (us)'].values[0])

# Calculate the required increase
required_increase = gdp_per_capita_france - gdp_per_capita_west_germany

# Since we need West Germany to surpass France, the increase must be more than this difference
# So the minimum increase is just above this value; we take the exact difference as the threshold
print(f"Final Answer: {required_increase:.0f}")