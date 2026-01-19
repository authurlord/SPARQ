import pandas as pd

df = pd.read_csv('table.csv')

# Extract numerical columns for correlation
numeric_cols = ['laps', 'grid', 'points']
correlations = df[numeric_cols].corr(numeric='pearson')

# Get correlations between points and laps, and points and grid
corr_laps = correlations.loc['points', 'laps']
corr_grid = correlations.loc['points', 'grid']

# Check if either correlation is significant (absolute value >= 0.3)
if abs(corr_laps) >= 0.3 or abs(corr_grid) >= 0.3:
    # Return the factors that have significant influence
    significant_factors = []
    if abs(corr_laps) >= 0.3:
        significant_factors.append('laps')
    if abs(corr_grid) >= 0.3:
        significant_factors.append('grid')
    Final Answer: , '.join(significant_factors)
else:
    Final Answer: no clear impact