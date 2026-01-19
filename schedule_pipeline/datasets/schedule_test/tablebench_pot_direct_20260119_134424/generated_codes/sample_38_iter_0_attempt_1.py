import pandas as pd

df = pd.read_csv('table.csv')
# Filter data for season 1
season_1_data = df[df['season'] == 1]
# Calculate average US viewers for season 1
avg_viewers_season_1 = season_1_data['us viewers (million)'].mean()
print(f"Final Answer: {avg_viewers_season_1:.2f}")