import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Convert 'season' to numeric (in case it's stored as string)
df['season'] = pd.to_numeric(df['season'], errors='coerce')

# Extract viewership for seasons 1 to 6 (inclusive)
seasons_1_to_6 = df[(df['season'] >= 1) & (df['season'] <= 6)]
viewership_1_to_6 = seasons_1_to_6['us viewers (millions)'].astype(float)

# Forecast viewership for season 7: take average of seasons 1 to 6
forecast_viewership = viewership_1_to_6.mean()

print(f"Final Answer: {forecast_viewership:.2f}")