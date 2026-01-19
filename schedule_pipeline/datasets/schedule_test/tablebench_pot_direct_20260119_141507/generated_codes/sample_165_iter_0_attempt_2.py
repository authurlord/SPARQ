import pandas as pd

df = pd.read_csv('table.csv')

# Compute correlation between each column and 'fraction'
correlations = df.corr()['fraction'].abs().sort_values(ascending=False)

# Get the top factor (with highest correlation)
top_factor = correlations.index[0]

# Check if the correlation is significant (greater than 0.3)
if correlations[top_factor] >= 0.3:
    print(f"Final Answer: {top_factor}")
else:
    print("Final Answer: no clear impact")