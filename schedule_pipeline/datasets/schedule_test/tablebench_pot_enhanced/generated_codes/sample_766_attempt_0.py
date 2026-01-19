import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')
# Convert 'total s ton' to numeric
df['total s ton'] = pd.to_numeric(df['total s ton'])

# Prepare data for linear regression
X = df['year'].astype(int).values.reshape(-1, 1)
y = df['total s ton'].values

# Fit linear regression model
model = np.polyfit(X.flatten(), y, 1)
predicted_2007 = np.polyval(model, 2007)

print(f"Final Answer: {int(predicted_2007)}")