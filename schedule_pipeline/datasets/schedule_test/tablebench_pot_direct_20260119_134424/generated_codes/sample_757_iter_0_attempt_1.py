import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Extract viewership data for seasons 1 to 6
viewership_seasons_1_to_6 = df[df['season'].between(1, 6)]['us viewers (millions)'].astype(float).values

# Create season numbers (1 to 6)
seasons = np.array([1, 2, 3, 4, 5, 6])

# Fit a linear trend line
slope, intercept = np.polyfit(seasons, viewership_seasons_1_to_6, 1)

# Forecast for season 7
forecast_season_7 = slope * 7 + intercept

print(f"Final Answer: {forecast_season_7:.2f}")