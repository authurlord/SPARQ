import pandas as pd

df = pd.read_csv('table.csv')

# Filter data for Suzuki in 50cc and 125cc classes
suzuki_50cc = df[(df['team'] == 'suzuki') & (df['class'] == '50cc')]
suzuki_125cc = df[(df['team'] == 'suzuki') & (df['class'] == '125cc')]

# Extract points and years
points_50cc = suzuki_50cc[['year', 'points']].astype(int)
points_125cc = suzuki_125cc[['year', 'points']].astype(int)

# Calculate average yearly increase for 50cc
if len(points_50cc) > 1:
    slope_50cc = (points_50cc['points'].iloc[-1] - points_50cc['points'].iloc[0]) / (int(points_50cc['year'].iloc[-1]) - int(points_50cc['year'].iloc[0]))
else:
    slope_50cc = 0

# Calculate average yearly increase for 125cc
if len(points_125cc) > 1:
    slope_125cc = (points_125cc['points'].iloc[-1] - points_125cc['points'].iloc[0]) / (int(points_125cc['year'].iloc[-1]) - int(points_125cc['year'].iloc[0]))
else:
    slope_125cc = 0

# Forecast for 1968
forecast_50cc = points_50cc['points'].iloc[-1] + slope_50cc
forecast_125cc = points_125cc['points'].iloc[-1] + slope_125cc

print(f"Final Answer: {forecast_50cc:.1f}, {forecast_125cc:.1f}")