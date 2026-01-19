import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Extract and convert data
df['broadcast date'] = pd.to_numeric(df['broadcast date'])
df['viewers (millions)'] = pd.to_numeric(df['viewers (millions)'])

# Filter data from 2005 to 2011 (episode 1 to 7)
data = df[df['broadcast date'] <= 2011]

# Prepare X (years) and y (viewers)
X = data['broadcast date'].values.reshape(-1, 1)
y = data['viewers (millions)'].values

# Fit linear regression model
model = np.polyfit(X.flatten(), y, 1)
predicted_2013 = np.polyval(model, 2013)

print(f"Final Answer: {predicted_2013:.2f}")