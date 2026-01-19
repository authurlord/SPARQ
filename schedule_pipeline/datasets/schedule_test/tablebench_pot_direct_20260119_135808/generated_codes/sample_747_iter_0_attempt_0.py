import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Convert year to numeric and urban percentage to float
df['year (january)'] = pd.to_numeric(df['year (january)'])
df['urban , %'] = pd.to_numeric(df['urban , %'])

# Prepare data for linear regression
X = df['year (january)'].values.reshape(-1, 1)
y = df['urban , %'].values

# Fit linear regression model
model = np.polyfit(X.flatten(), y, 1)
predicted_urban_percentage = np.polyval(model, 2015)

print(f"Final Answer: {predicted_urban_percentage:.1f}")