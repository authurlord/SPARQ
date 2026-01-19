import pandas as pd

df = pd.read_csv('table.csv')

# Select relevant columns for correlation analysis
columns_to_analyze = ['played', 'drawn', 'lost', 'points for', 'points against', 'tries for', 'tries against', 'try bonus', 'losing bonus']
correlations = df[columns_to_analyze].corrwith(df['points'])

# Get top 3 factors by absolute correlation
top_3_factors = correlations.abs().sort_values(ascending=False).head(3).index.tolist()

print(f"Final Answer: {', '.join(top_3_factors)}")