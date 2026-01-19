import pandas as pd

df = pd.read_csv('table.csv')

# Convert population and GDP per capita to numeric (in case of string)
df['population'] = pd.to_numeric(df['population'], errors='coerce')
df['gdp per capita (us)'] = pd.to_numeric(df['gdp per capita (us)'], errors='coerce')

# Remove rows with missing values
df = df.dropna(subset=['gdp per capita (us)'])

# Calculate mean and std of GDP per capita
mean_gdp_per_capita = df['gdp per capita (us)'].mean()
std_gdp_per_capita = df['gdp per capita (us)'].std()

# Identify outliers (more than 2 std from mean)
lower_bound = mean_gdp_per_capita - 2 * std_gdp_per_capita
upper_bound = mean_gdp_per_capita + 2 * std_gdp_per_capita

outliers = df[(df['gdp per capita (us)'] < lower_bound) | (df['gdp per capita (us)'] > upper_bound)]
outlier_countries = outliers['member countries'].tolist()

print(f"Final Answer: {', '.join(outlier_countries)}")