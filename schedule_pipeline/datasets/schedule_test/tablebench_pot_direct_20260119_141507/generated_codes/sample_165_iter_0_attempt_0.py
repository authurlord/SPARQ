import pandas as pd

df = pd.read_csv('table.csv')
# Select columns for correlation analysis
columns_to_check = ['half - life (s)', 'decay constant (s 1)', 'yield , neutrons per fission']
correlations = df[columns_to_check + ['fraction']].corr()['fraction'].abs()

# Find which factors have a correlation greater than 0.7 (considered significant)
significant_factors = [col for col in columns_to_check if correlations[col] > 0.7]

if significant_factors:
    print(f"Final Answer: {', '.join(significant_factors)}")
else:
    print("Final Answer: no clear impact")