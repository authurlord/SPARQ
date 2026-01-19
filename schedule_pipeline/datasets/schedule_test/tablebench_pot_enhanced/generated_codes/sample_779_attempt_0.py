import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Convert 'bötzow' column to float
df['bötzow'] = df['bötzow'].astype(float)

# Extract year and bötzow data
years = df['year'].values
pop_bötzow = df['bötzow'].values

# Perform linear regression to predict 2015
X = years.reshape(-1, 1)
y = pop_bötzow
model = np.polyfit(X.flatten(), y, 1)
predicted_2015 = np.polyval(model, 2015)

# Plot the trend
plt.plot(years, pop_bötzow, marker='o', label='Bötzow Population')
plt.axvline(x=2015, color='r', linestyle='--', label='Predicted 2015')
plt.xlabel('Year')
plt.ylabel('Population (in thousands)')
plt.title('Population Trend of Bötzow and Prediction for 2015')
plt.legend()
plt.grid(True)
plt.show()

print(f"Final Answer: {predicted_2015:.3f}")