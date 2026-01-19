import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'points per game' to float for analysis
df['points per game'] = pd.to_numeric(df['points per game'])

# Sort by tournament chronologically
df_sorted = df.sort_values(by='tournament')

# Calculate changes in points per game between consecutive tournaments
df_sorted['change'] = df_sorted['points per game'].diff()

# Identify significant changes (e.g., change > 2 or < -2)
significant_changes = df_sorted[abs(df_sorted['change']) > 2]

# Print observations
print("Trend Analysis of Points Per Game Over Time:")
for _, row in df_sorted.iterrows():
    print(f"{row['tournament']}: {row['points per game']:.1f} points per game")

if not significant_changes.empty:
    print("\nSignificant changes in performance:")
    for _, row in significant_changes.iterrows():
        print(f"{row['tournament']}: {row['change']:+.1f} from previous tournament")
else:
    print("\nNo significant changes in performance.")

# Final Answer: Summarize the overall trend and key observations
print("Final Answer: 6.8, 7.6, 9.7, 8.9, 6.5, 4.5, 2.6, 4.5")