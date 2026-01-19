import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Convert year to numeric for modeling
df['year (january)'] = pd.to_numeric(df['year (january)'])

# Extract features and target
X = df['year (january)'].values.reshape(-1, 1)
y = df['urban , %'].values

# Fit linear regression model
model = np.polyfit(X.flatten(), y, 1)
predicted_2015 = np.polyval(model, 2015)

print(f"Final Answer: {predicted_2015:.1f}")