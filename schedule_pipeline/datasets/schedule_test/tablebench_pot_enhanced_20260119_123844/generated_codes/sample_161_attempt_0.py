import pandas as pd

df = pd.read_csv('table.csv')

# Convert 'total viewers' to numeric
df['total viewers'] = pd.to_numeric(df['total viewers'], errors='coerce')

# Convert 'share' to numeric by removing '%' and converting
df['share'] = pd.to_numeric(df['share'].str.replace('%', ''), errors='coerce')

# Convert 'bbc one weekly ranking' to numeric
df['bbc one weekly ranking'] = pd.to_numeric(df['bbc one weekly ranking'], errors='coerce')

# Calculate correlation matrix
correlation_matrix = df[['total viewers', 'share', 'bbc one weekly ranking']].corr()

# Extract correlations with 'bbc one weekly ranking'
corr_with_ranking = correlation_matrix['bbc one weekly ranking'].drop('bbc one weekly ranking')

# Check if any correlation is strong (absolute value > 0.7)
if corr_with_ranking.abs().max() > 0.7:
    # Find the factor with highest absolute correlation
    strongest_factor = corr_with_ranking.abs().idxmax()
    print(f"Final Answer: {strongest_factor}")
else:
    print("Final Answer: no clear impact")