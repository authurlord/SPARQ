import pandas as pd

df = pd.read_csv('table.csv')

# Select numerical columns for correlation analysis
numerical_cols = ['races', 'wins', 'poles', 'laps', 'podiums']
correlations = df[numerical_cols + ['points']].corr()['points'].abs()

# Threshold for significance (e.g., > 0.5)
significant_factors = [col for col, corr_val in correlations.items() if col != 'points' and corr_val > 0.5]

# Check for categorical patterns
# For 'series' and 'team', check if high points correlate with specific entries
high_points_series = df[df['points'] == df['points'].max()]['series'].values
high_points_team = df[df['points'] == df['points'].max()]['team'].values

# If any numerical factor has strong correlation or categorical entries show clear trends, report them
if significant_factors:
    final_factors = significant_factors
else:
    final_factors = ['no clear impact']

print(f"Final Answer: {', '.join(final_factors)}")