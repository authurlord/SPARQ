import pandas as pd

df = pd.read_csv('table.csv')
# Calculate correlation with market value
correlations = df[['sales (billion )', 'profits (billion )', 'assets (billion )', 'market value (billion )']].corr()['market value (billion )']
# Find the factor with the highest absolute correlation
main_factor = correlations.abs().idxmax()
print(f"Final Answer: {main_factor}")