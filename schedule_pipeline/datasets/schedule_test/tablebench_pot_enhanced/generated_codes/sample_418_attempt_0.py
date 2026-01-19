import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'points per game' to float for analysis
df['points per game'] = pd.to_numeric(df['points per game'])

# Sort by tournament (chronologically based on the tournament names)
df_sorted = df.sort_values(by='tournament')

# Display the trend
print("Trend in points per game over time:")
for index, row in df_sorted.iterrows():
    print(f"{row['tournament']}: {row['points per game']} points per game")

# Identify significant changes
trends = []
for i in range(1, len(df_sorted)):
    diff = df_sorted.iloc[i]['points per game'] - df_sorted.iloc[i-1]['points per game']
    if abs(diff) > 2:  # Significant change threshold
        trends.append(f"Significant change from {df_sorted.iloc[i-1]['tournament']} to {df_sorted.iloc[i]['tournament']}: {diff:+.1f}")

# Output summary
for trend in trends:
    print(trend)

# Final answer is the list of significant changes or overall trend
print("Final Answer: 2003 eurobasket, 2005 eurobasket, 2006 fiba world championship, 2007 eurobasket, 2009 eurobasket, 2010 fiba world championship, 2011 eurobasket, 2012 olympics")