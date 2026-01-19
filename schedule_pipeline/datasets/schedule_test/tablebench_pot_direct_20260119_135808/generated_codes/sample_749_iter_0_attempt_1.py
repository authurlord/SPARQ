import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Convert 'broadcast date' to integer year and 'viewers (millions)' to float
df['broadcast date'] = pd.to_numeric(df['broadcast date'])
df['viewers (millions)'] = pd.to_numeric(df['viewers (millions)'])

# Select data from 2005 to 2011 (episode 1 to 7)
df_trend = df[df['broadcast date'] <= 2011]

# Prepare data for linear regression
X = df_trend['broadcast date'].values.reshape(-1, 1)
y = df_trend['viewers (millions)'].values

# Fit linear regression model
model = np.polyfit(X.flatten(), y, 1)
predicted_2013 = np.polyval(model, 2013)

print(f"Final Answer: {predicted_2013:.2f}")