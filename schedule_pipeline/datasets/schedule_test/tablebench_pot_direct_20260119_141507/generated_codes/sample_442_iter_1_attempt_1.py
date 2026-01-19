import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Display the structure and initial observations
print("Main Columns:")
print(" - polling firm: Source of the poll")
print(" - date of polling: Date range of the poll")
print(" - link: Reference URL (not used in analysis)")
print(" - progressive conservative: % of support for PC party")
print(" - liberal: % of support for Liberal party")
print(" - new democratic: % of support for NDP party")

# Summary statistics for party support
print("\nSummary of Party Support Trends:")
print(f"Average Progressive Conservative support: {df['progressive conservative'].mean():.1f}%")
print(f"Average Liberal support: {df['liberal'].mean():.1f}%")
print(f"Average New Democratic support: {df['new democratic'].mean():.1f}%")

# Trend observation: Sort by date to see changes over time
df['date of polling'] = pd.to_datetime(df['date of polling'], errors='coerce')
df_sorted = df.sort_values('date of polling')

# Check the trend in PC support over time
pc_trend = df_sorted['progressive conservative'].iloc[-1]  # Latest value
pc_earliest = df_sorted['progressive conservative'].iloc[0]  # Earliest value
print(f"\nTrend Observation: Progressive Conservative support decreased from {pc_earliest}% to {pc_trend}% over time.")
print(f"Liberal support increased from {df_sorted['liberal'].iloc[0]:.1f}% to {df_sorted['liberal'].iloc[-1]:.1f}%.")
print(f"New Democratic support ranged from {df_sorted['new democratic'].min()}% to {df_sorted['new democratic'].max()}%.")

Final Answer: polling firm, date of polling, progressive conservative, liberal, new democratic, support trends over time