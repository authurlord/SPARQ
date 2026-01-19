import pandas as pd

df = pd.read_csv('table.csv')

# Select only numerical columns for correlation analysis
numerical_columns = ['population (2011)', 'population (2006)', 'area (km square)', 'population density']
# Convert to numeric, handling any non-numeric entries
df_numeric = df[numerical_columns].apply(pd.to_numeric, errors='coerce')

# Drop rows with NaN due to conversion issues
df_clean = df_numeric.dropna()

# Compute correlation between 'change (%)' and each numerical column
correlations = df_clean['change (%)'].corr(df_clean['population (2011)']), \
               df_clean['change (%)'].corr(df_clean['population (2006)']), \
               df_clean['change (%)'].corr(df_clean['area (km square)']), \
               df_clean['change (%)'].corr(df_clean['population density'])

# Check if any correlation has absolute value > 0.3 (considered significant)
significant_factors = []
for i, col in enumerate(['population (2011)', 'population (2006)', 'area (km square)', 'population density']):
    if abs(correlations[i]) > 0.3:
        significant_factors.append(col)

if significant_factors:
    print(f"Final Answer: {', '.join(significant_factors)}")
else:
    print("Final Answer: no clear impact")