import pandas as pd

df = pd.read_csv('table.csv')

# Calculate average number of tropical storms and hurricanes per year
avg_storms = df['number of tropical storms'].astype(int).mean()
avg_hurricanes = df['number of hurricanes'].astype(int).mean()

# Forecast for next 5 years using average values
forecast_storms = avg_storms * 5
forecast_hurricanes = avg_hurricanes * 5

print(f"Final Answer: {avg_storms:.1f}, {avg_hurricanes:.1f}")