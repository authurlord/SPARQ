import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Drop the first row (header row) if it's duplicated or causing issues
# The first row is actually a header, so we keep it and ensure proper parsing
# Convert all relevant columns to numeric, handling any potential errors
df = df.iloc[1:]  # Skip the first row which contains column names

# Convert relevant columns to numeric type
numeric_columns = ['played', 'drawn', 'lost', 'points for', 'points against', 'tries for', 'tries against', 'try bonus', 'losing bonus', 'points']
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows with NaN after conversion
df = df.dropna()

# Compute correlation between each factor and 'points'
correlations = df.corr()['points'].abs().sort_values(ascending=False)

# Get top 3 factors
top_3_factors = correlations.head(3).index.tolist()

print(f"Final Answer: {top_3_factors[0]}, {top_3_factors[1]}, {top_3_factors[2]}")