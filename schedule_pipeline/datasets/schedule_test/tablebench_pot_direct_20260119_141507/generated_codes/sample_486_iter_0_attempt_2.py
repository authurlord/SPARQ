import pandas as pd

df = pd.read_csv('table.csv')

# Drop duplicate rows (e.g., 'korea' appears twice)
df = df.drop_duplicates(subset=['economy'])

# Calculate average gap from Thailand as of 2012
avg_gap_2012 = df['gap from thailand as of 2012 (times)'].mean()

# Define threshold for significant deviation (e.g., more than 1.5 times average)
threshold = avg_gap_2012 * 1.5

# Identify economies with gap significantly above or below average
high_deviation = df[df['gap from thailand as of 2012 (times)'] > threshold]
low_deviation = df[df['gap from thailand as of 2012 (times)'] < avg_gap_2012 * 0.5]

# Combine and get economy names
deviant_economies = high_deviation['economy'].tolist() + low_deviation['economy'].tolist()

# Remove duplicates
deviant_economies = list(set(deviant_economies))

# Also check GDP per capita for extreme values (outliers)
gdp_per_capita = df['gdp per capita as of 2012 (ppp)']
gdp_max = gdp_per_capita.max()
gdp_min = gdp_per_capita.min()
gdp_threshold = (gdp_max - gdp_min) * 0.5

# Add economies with GDP per capita far from average
gdp_outliers = df[(gdp_per_capita > gdp_max - gdp_threshold) | (gdp_per_capita < gdp_min + gdp_threshold)]
gdp_outlier_economies = gdp_outliers['economy'].tolist()

# Final list of deviant economies (union of gap and GDP outliers)
final_deviant = list(set(deviant_economies + gdp_outlier_economies))

# Print result
print(f"Final Answer: {', '.join(final_deviant)}")