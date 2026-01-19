import pandas as pd

df = pd.read_csv('table.csv')

# Calculate correlation between points and laps, and points and grid
correlation_laps = df['laps'].corr(df['points'])
correlation_grid = df['grid'].corr(df['points'])

# Group by team and calculate average points per team
team_points = df.groupby('team')['points'].mean()

# Check if any correlation is significant (absolute value > 0.3)
if abs(correlation_laps) > 0.3 or abs(correlation_grid) > 0.3:
    # If significant correlation, return the factor(s)
    if abs(correlation_laps) > 0.3:
        result = "laps"
    elif abs(correlation_grid) > 0.3:
        result = "grid"
    else:
        result = "no clear impact"
else:
    # No significant correlation
    result = "no clear impact"

print(f"Final Answer: {result}")