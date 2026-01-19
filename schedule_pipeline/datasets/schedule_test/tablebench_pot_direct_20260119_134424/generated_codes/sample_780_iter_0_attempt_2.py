import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')
# Convert 'total support and revenue' to numeric
df['total support and revenue'] = pd.to_numeric(df['total support and revenue'])

# Create year index (e.g., 2003/2004 -> 1, 2004/2005 -> 2, etc.)
df['year_index'] = range(1, len(df) + 1)

# Perform linear regression to find trend
X = df['year_index'].values.reshape(-1, 1)
y = df['total support and revenue'].values
model = np.polyfit(X.flatten(), y, 1)
predicted_next_year = model[0] * (len(df) + 1) + model[1]

print(f"Final Answer: {predicted_next_year:.0f}")