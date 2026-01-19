import pandas as pd

df = pd.read_csv('table.csv')

# Convert 'population' and 'gdp per capita (us)' to numeric, coercing errors to NaN
df['population'] = pd.to_numeric(df['population'], errors='coerce')
df['gdp per capita (us)'] = pd.to_numeric(df['gdp per capita (us)'], errors='coerce')

# Drop rows where population or gdp per capita is NaN
df_clean = df.dropna(subset=['population', 'gdp per capita (us)'])

# Function to detect outliers using IQR
def detect_outliers(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return series[(series < lower_bound) | (series > upper_bound)]

# Detect outliers in population and gdp per capita
outliers_population = detect_outliers(df_clean['population'])
outliers_gdp_per_capita = detect_outliers(df_clean['gdp per capita (us)'])

# Get unique countries with outliers
outlier_countries = set(outliers_population.index).union(set(outliers_gdp_per_capita.index))
outlier_countries_names = df_clean.loc[outlier_countries, 'member countries'].tolist()

print(f"Final Answer: {', '.join(outlier_countries_names)}")