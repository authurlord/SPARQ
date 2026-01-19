import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Convert 'pass percentage' to float by removing '%' and converting to float
df['pass percentage'] = df['pass percentage'].str.replace('%', '').astype(float)

# Convert 'obtained gpa - 5' to numeric
df['obtained gpa - 5'] = pd.to_numeric(df['obtained gpa - 5'])

# Prepare data for linear regression
X = df['year'].astype(int).values.reshape(-1, 1)
y_pass = df['pass percentage'].values
y_gpa = df['obtained gpa - 5'].values

# Fit linear models
model_pass = np.polyfit(X.flatten(), y_pass, 1)
model_gpa = np.polyfit(X.flatten(), y_gpa, 1)

# Predict for 2011
year_2011 = np.array([[2011]])
pred_pass = np.polyval(model_pass, 2011)
pred_gpa = np.polyval(model_gpa, 2011)

print(f"Final Answer: {pred_pass:.2f}, {pred_gpa:.2f}")