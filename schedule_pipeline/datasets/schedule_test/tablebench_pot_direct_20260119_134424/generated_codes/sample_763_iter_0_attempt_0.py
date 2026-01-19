import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')
# Convert 'bötzow' to numeric
df['bötzow'] = pd.to_numeric(df['bötzow'])

# Create year as a numerical feature
df['year_num'] = df['year'].astype(int)

# Fit a linear regression model
X = df['year_num'].values.reshape(-1, 1)
y = df['bötzow'].values
model = np.polyfit(X.flatten(), y, 1)
predicted_values = np.polyval(model, [2011, 2012])

# Output the predictions
print(f"Final Answer: {predicted_values[0]:.3f}, {predicted_values[1]:.3f}")