import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')
df['qual'] = pd.to_numeric(df['qual'], errors='coerce')

# Prepare data for linear regression
X = df['year'].astype(int).values.reshape(-1, 1)
y = df['qual'].values

# Fit linear regression model
model = np.polyfit(X.flatten(), y, 1)
predicted_qual_1960 = np.polyval(model, 1960)

print(f"Final Answer: {predicted_qual_1960:.3f}")