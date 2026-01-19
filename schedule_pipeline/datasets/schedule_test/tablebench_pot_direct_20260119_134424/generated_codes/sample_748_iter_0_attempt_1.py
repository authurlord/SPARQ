import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')
# Convert 'year' to integer and 'qual' to float
df['year'] = df['year'].astype(int)
df['qual'] = df['qual'].astype(float)

# Perform linear regression: y = mx + b, where y is 'qual', x is 'year'
X = df['year'].values.reshape(-1, 1)
y = df['qual'].values

# Fit the linear model
model = np.polyfit(X.flatten(), y, 1)
predicted_qual_1960 = np.polyval(model, 1960)

print(f"Final Answer: {predicted_qual_1960:.3f}")