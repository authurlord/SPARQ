import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Convert 'change (%)' to numeric, handling any parsing issues
df['change (%)'] = pd.to_numeric(df['change (%)'], errors='coerce')

# Drop rows with missing 'change (%)' values
df = df.dropna(subset=['change (%)'])

# Select relevant numerical columns
numeric_columns = ['area (km square)', 'population density']
correlations = df[numeric_columns].corrwith(df['change (%)'])

# Check if any correlation is above 0.3 in absolute value
significant_correlations = correlations[abs(correlations) > 0.3]

if significant_correlations.empty:
    print("Final Answer: no clear impact")
else:
    # Return the names of factors with significant correlation
    significant_factors = list(significant_correlations.index)
    print(f"Final Answer: {', '.join(significant_factors)}")