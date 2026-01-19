import pandas as pd

df = pd.read_csv('table.csv')
# Extract GDP per capita column and convert to numeric
gdp_per_capita = pd.to_numeric(df['gdp per capita (us)'], errors='coerce')

# Calculate mean and std for outlier detection
mean_gdp_per_capita = gdp_per_capita.mean()
std_gdp_per_capita = gdp_per_capita.std()

# Define outlier threshold (2 standard deviations)
threshold = 2 * std_gdp_per_capita
outliers = gdp_per_capita[(gdp_per_capita > mean_gdp_per_capita + threshold) | 
                          (gdp_per_capita < mean_gdp_per_capita - threshold)]

# Get the corresponding country names
outlier_countries = df[df['gdp per capita (us)'].isin(outliers)].iloc[:, 0].tolist()

print(f"Final Answer: {', '.join(outlier_countries)}")