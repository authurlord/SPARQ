import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')
# Convert 'bötzow' column to float
df['bötzow'] = df['bötzow'].astype(float)

# Extract year and bötzow data
years = df['year'].astype(int)
population = df['bötzow']

# Fit a linear regression model
coefficients = np.polyfit(years, population, 1)
poly_model = np.poly1d(coefficients)

# Predict population for 2015
predicted_2015 = poly_model(2015)

# Plot the trend
plt.figure(figsize=(10, 5))
plt.plot(years, population, marker='o', label='Bötzow Population')
plt.axvline(x=2015, color='r', linestyle='--', label='Prediction Year (2015)')
plt.xlabel('Year')
plt.ylabel('Population (in thousands)')
plt.title('Population Trend of Bötzow and Prediction for 2015')
plt.legend()
plt.grid(True)
plt.show()

print(f"Final Answer: {predicted_2015:.3f}")