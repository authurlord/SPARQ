import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')
df['year'] = pd.to_numeric(df['year'])
df['qual'] = pd.to_numeric(df['qual'])

# Fit a linear regression model
X = df['year'].values.reshape(-1, 1)
y = df['qual'].values
model = np.polyfit(X.flatten(), y, 1)
predicted_qual_1960 = np.polyval(model, 1960)

print(f"Final Answer: {predicted_qual_1960:.3f}")