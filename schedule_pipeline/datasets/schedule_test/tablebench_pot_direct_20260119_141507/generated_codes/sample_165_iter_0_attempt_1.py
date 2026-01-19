import pandas as pd

df = pd.read_csv('table.csv')
# Compute correlation between each feature and 'fraction'
correlations = df[['half - life (s)', 'decay constant (s 1)', 'yield , neutrons per fission', 'fraction']].corr()['fraction'].abs()
# Find the factor with the highest correlation
top_correlation = correlations.idxmax()
top_corr_value = correlations.max()

# Check if the highest correlation is strong (greater than 0.5)
if top_corr_value >= 0.5:
    print(f"Final Answer: {top_correlation}")
else:
    print("Final Answer: no clear impact")