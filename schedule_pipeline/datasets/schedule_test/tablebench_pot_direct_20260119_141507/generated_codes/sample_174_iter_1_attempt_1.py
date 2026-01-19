import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Convert 'change (%)' to numeric (handle strings like '5.9', '-0.4')
df['change (%)'] = pd.to_numeric(df['change (%)'], errors='coerce')

# Select only numeric columns for correlation
numeric_columns = ['area (km square)', 'population density']
# We exclude 'population (2011)' and 'population (2006)' since they are redundant with 'change (%)'

# Calculate correlation with 'change (%)'
correlations = df[numeric_columns].corrwith(df['change (%)'])

# Find columns with absolute correlation > 0.3 (heuristic for significant influence)
significant_factors = []
for col in numeric_columns:
    if abs(correlations[col]) > 0.3:
        significant_factors.append(col)

if significant_factors:
    print(f"Final Answer: {', '.join(significant_factors)}")
else:
    print("Final Answer: no clear impact")