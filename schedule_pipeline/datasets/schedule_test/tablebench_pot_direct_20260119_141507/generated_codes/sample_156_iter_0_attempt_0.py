import pandas as pd

df = pd.read_csv('table.csv')

# Select relevant columns
correlations = df[['population % of eu', 'area km 2', 'area % of eu', 'pop density people / km 2']].corr(numeric_only=True)

# Extract correlation with 'population % of eu'
corr_area_km2 = correlations['population % of eu']['area km 2']
corr_area_percent = correlations['population % of eu']['area % of eu']
corr_pop_density = correlations['population % of eu']['pop density people / km 2']

# Check if any correlation has absolute value > 0.3 (considered significant)
significant_factors = []
if abs(corr_area_km2) > 0.3:
    significant_factors.append('area km 2')
if abs(corr_area_percent) > 0.3:
    significant_factors.append('area % of eu')
if abs(corr_pop_density) > 0.3:
    significant_factors.append('pop density people / km 2')

if significant_factors:
    print(f"Final Answer: {', '.join(significant_factors)}")
else:
    print("Final Answer: no clear impact")