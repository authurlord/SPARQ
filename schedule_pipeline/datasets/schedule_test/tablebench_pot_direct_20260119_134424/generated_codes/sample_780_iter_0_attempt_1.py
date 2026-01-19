import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')
# Convert 'total support and revenue' to numeric
df['total support and revenue'] = pd.to_numeric(df['total support and revenue'])

# Extract the year and convert to numeric (e.g., 2003/2004 → 2003)
df['year_numeric'] = df['year'].str.extract(r'(\d{4})').astype(int)

# Perform linear regression to predict the next year
X = df['year_numeric'].values.reshape(-1, 1)
y = df['total support and revenue'].values

# Fit linear model
model = np.polyfit(X.flatten(), y, 1)
predicted_next_year = model[0] * (df['year_numeric'].max() + 1) + model[1]

print(f"Final Answer: {int(predicted_next_year)}")