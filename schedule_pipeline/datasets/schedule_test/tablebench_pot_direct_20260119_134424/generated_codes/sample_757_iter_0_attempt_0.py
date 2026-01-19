import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Filter data for seasons 1 to 6
seasons_1_to_6 = df[df['season'].between(1, 6)]
x = seasons_1_to_6['season'].values
y = seasons_1_to_6['us viewers (millions)'].values

# Fit a linear regression model
coefficients = np.polyfit(x, y, 1)
trend_line = np.poly1d(coefficients)

# Predict viewership for season 7
predicted_season_7 = trend_line(7)

# Plot the data and trend line for visualization
plt.scatter(x, y, color='blue', label='Actual Viewership')
plt.plot(x, trend_line(x), color='red', linestyle='--', label='Trend Line')
plt.scatter(7, predicted_season_7, color='green', label='Predicted Season 7')
plt.xlabel('Season')
plt.ylabel('US Viewers (millions)')
plt.title('Viewership Trend and Forecast for Season 7')
plt.legend()
plt.grid(True)
plt.show()

print(f"Final Answer: {predicted_season_7:.2f}")