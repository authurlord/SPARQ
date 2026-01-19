import pandas as pd

# Load the data
df = pd.read_csv('table.csv', header=0)

# Drop the first row if it's a duplicate header (it's actually a header row)
# The first row is already headers, so we don't need to drop anything
# Ensure correct data types
df = df.drop(df.index[0])  # Remove the first row which is just the header duplicated

# Convert relevant columns to numeric, handling any parsing issues
numeric_columns = ['played', 'drawn', 'lost', 'points for', 'points against', 'tries for', 'tries against', 'try bonus', 'losing bonus', 'points']
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows with NaN values due to conversion
df = df.dropna()

# Compute correlation between each factor and 'points'
correlations = df.corr()['points'].abs().sort_values(ascending=False)

# Get top 3 factors
top3_factors = correlations.head(3).index.tolist()

print(f"Final Answer: {top3_factors[0]}, {top3_factors[1]}, {top3_factors[2]}")