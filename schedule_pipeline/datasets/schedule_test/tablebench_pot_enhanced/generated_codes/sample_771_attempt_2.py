import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Load the data
df = pd.read_csv('table.csv')

# Convert 'pass percentage' to numeric by removing '%' and converting to float
df['pass percentage'] = df['pass percentage'].str.replace('%', '').astype(float)

# Prepare the data for modeling
X = df['year'].values.reshape(-1, 1)  # Year as feature
y_pass = df['pass percentage'].values  # Target: pass percentage
y_gpa = df['obtained gpa - 5'].values  # Target: GPA

# Fit linear regression models
model_pass = LinearRegression().fit(X, y_pass)
model_gpa = LinearRegression().fit(X, y_gpa)

# Predict for 2011
year_2011 = np.array([[2011]])
predicted_pass = model_pass.predict(year_2011)[0]
predicted_gpa = model_gpa.predict(year_2011)[0]

print(f"Final Answer: {predicted_pass:.2f}, {int(predicted_gpa)}")