import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Drop the first row which contains column headers (it's duplicated)
# The first row is actually the header, so we don't need to drop it — the column names are already set correctly
# But the first data row is a club name, so we skip the first row if it's just headers
# In this case, the first row is actually a header row with club names, so we start from index 1
df = df.iloc[1:]  # Skip the first row (which is the header)

# Convert relevant columns to numeric, handling any potential issues
numeric_columns = ['played', 'drawn', 'lost', 'points for', 'points against', 'tries for', 'tries against', 'try bonus', 'losing bonus']
df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

# Drop rows with NaN values after conversion
df = df.dropna()

# Compute correlation with 'points'
correlations = df[numeric_columns].corrwith(df['points'])

# Get top 3 factors with highest absolute correlation
top_3_factors = correlations.abs().sort_values(ascending=False).head(3).index.tolist()

print(f"Final Answer: {', '.join(top_3_factors)}")