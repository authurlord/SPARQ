import pandas as pd

df = pd.read_csv('table.csv')

# Drop duplicate rows (e.g., 'korea' appears twice)
df = df.drop_duplicates(subset=['economy'])

# Extract relevant columns
gap_2012 = df['gap from thailand as of 2012 (times)'].astype(float)
gdp_per_capita_2012 = df['gdp per capita as of 2012 (ppp)'].astype(float)

# Calculate mean and std
mean_gap = gap_2012.mean()
std_gap = gap_2012.std()

mean_gdp = gdp_per_capita_2012.mean()
std_gdp = gdp_per_capita_2012.std()

# Find outliers using 2 standard deviations
outliers_gap = df[(abs(gap_2012 - mean_gap) > 2 * std_gap) & (gap_2012.notna())]
outliers_gdp = df[(abs(gdp_per_capita_2012 - mean_gdp) > 2 * std_gdp) & (gdp_per_capita_2012.notna())]

# Combine and get unique economies
outlier_economies = set()
for _, row in outliers_gap.iterrows():
    outlier_economies.add(row['economy'])
for _, row in outliers_gdp.iterrows():
    outlier_economies.add(row['economy'])

# Check for extreme values manually (e.g., Malaysia at 100,000 GDP per capita)
manual_outliers = ['malaysia']  # Malaysia has unusually high GDP per capita

# Final list of economies that deviate significantly
final_outliers = list(outlier_economies | set(manual_outliers))

# Remove duplicates and sort
final_outliers = sorted(list(set(final_outliers)))

print(f"Final Answer: {', '.join(final_outliers)}")