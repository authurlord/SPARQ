import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'total capacity (mw)' to numeric
df['total capacity (mw)'] = pd.to_numeric(df['total capacity (mw)'])
# Convert 'completion schedule' to integer
df['completion schedule'] = pd.to_numeric(df['completion schedule'])

# Calculate correlation between capacity and completion year
correlation = df['total capacity (mw)'].corr(df['completion schedule'])

# Print result with explanation
print(f"Correlation between capacity and completion year: {correlation:.3f}")
print("Note: Correlation does not imply causation. No causal relationship can be concluded from this data alone.")