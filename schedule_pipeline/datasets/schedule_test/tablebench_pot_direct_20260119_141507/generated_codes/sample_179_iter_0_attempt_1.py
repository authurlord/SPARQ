import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Select only numeric columns for correlation
numeric_cols = ['races', 'wins', 'poles', 'laps', 'podiums', 'points']
df_numeric = df[numeric_cols]

# Compute correlation with 'points'
correlations = df_numeric.corr()['points'].abs().sort_values(ascending=False)

# Identify factors with correlation > 0.3
significant_factors = []
for col in correlations.index:
    if correlations[col] > 0.3:
        significant_factors.append(col)

if significant_factors:
    final_answer = ', '.join(significant_factors)
else:
    final_answer = 'no clear impact'

print(f"Final Answer: {final_answer}")