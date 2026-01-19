import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Convert year to numeric and extract urban percentage
df['year'] = pd.to_numeric(df['year (january)'])
urban_percent = df['urban , %'].astype(float)

# Perform linear regression
X = df['year'].values.reshape(-1, 1)
y = urban_percent.values

# Fit linear model
model = np.polyfit(X.flatten(), y, 1)
predicted_2015 = np.polyval(model, 2015)

print(f"Final Answer: {predicted_2015:.1f}")