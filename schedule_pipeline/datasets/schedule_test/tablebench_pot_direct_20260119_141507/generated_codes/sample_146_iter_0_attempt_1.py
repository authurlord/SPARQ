import pandas as pd

df = pd.read_csv('table.csv')
# Compute correlation between each factor and 'points'
correlations = df.drop(columns=['club', 'played', 'drawn', 'lost', 'tries for', 'tries against', 'try bonus', 'losing bonus']).corrwith(df['points'])
# Get top 3 factors by absolute correlation
top_3_factors = correlations.abs().sort_values(ascending=False).head(3).index.tolist()
print(f"Final Answer: {', '.join(top_3_factors)}")