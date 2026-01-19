import pandas as pd

df = pd.read_csv('table.csv')
# Filter data for Season 1
season_1_data = df[df['season'] == 1]
# Calculate average US viewers for Season 1
average_viewers = season_1_data['us viewers (million)'].mean()
print(f"Final Answer: {average_viewers:.2f}")