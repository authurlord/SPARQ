import pandas as pd

df = pd.read_csv('table.csv')
# Extract viewership for seasons 1 to 6
seasons_1_to_6 = df[df['season'].between(1, 6)]
viewership_1_to_6 = seasons_1_to_6['us viewers (millions)']
# Forecast as the average of seasons 1 to 6
forecast_season_7 = viewership_1_to_6.mean()
print(f"Final Answer: {forecast_season_7:.2f}")