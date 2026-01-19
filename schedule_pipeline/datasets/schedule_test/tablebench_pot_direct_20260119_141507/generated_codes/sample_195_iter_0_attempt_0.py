import pandas as pd

df = pd.read_csv('table.csv')

# Select only numerical columns for correlation
numeric_columns = ['laps', 'grid', 'points']
correlation_matrix = df[numeric_columns].corr()

# Extract correlation between 'points' and 'laps', and 'points' and 'grid'
corr_laps = correlation_matrix.loc['points', 'laps']
corr_grid = correlation_matrix.loc['points', 'grid']

# Check if either correlation is strong (absolute value > 0.3)
if abs(corr_laps) > 0.3 or abs(corr_grid) > 0.3:
    # Return the factor(s) with significant correlation
    if abs(corr_laps) > 0.3:
        significant_factor = 'laps'
    elif abs(corr_grid) > 0.3:
        significant_factor = 'grid'
    else:
        significant_factor = 'no clear impact'
else:
    significant_factor = 'no clear impact'

print(f"Final Answer: {significant_factor}")