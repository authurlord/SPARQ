import pandas as pd

df = pd.read_csv('table.csv')

# Select only numeric columns for correlation
numeric_columns = ['races', 'wins', 'poles', 'laps', 'podiums', 'points']
correlation_df = df[numeric_columns].corr()

# Extract correlation between 'points' and other variables
points_corr = correlation_df['points'].drop('points').abs()

# Identify factors with correlation > 0.3 (considered significant)
significant_factors = points_corr[points_corr > 0.3].index.tolist()

if len(significant_factors) == 0:
    print("Final Answer: no clear impact")
else:
    print(f"Final Answer: {', '.join(significant_factors)}")