import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')
# Convert 'year' to integer and 'qual' to float
df['year'] = pd.to_numeric(df['year'])
df['qual'] = pd.to_numeric(df['qual'])

# Prepare data for linear regression
X = df['year'].values.reshape(-1, 1)
y = df['qual'].values

# Fit linear regression model
model = np.polyfit(X.flatten(), y, 1)
predicted_speed_1960 = np.polyval(model, 1960)

print(f"Final Answer: {predicted_speed_1960:.3f}")