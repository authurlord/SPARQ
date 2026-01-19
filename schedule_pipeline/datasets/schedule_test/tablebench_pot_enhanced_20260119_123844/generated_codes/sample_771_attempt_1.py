import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Convert 'pass percentage' to numeric by removing '%' and converting to float
df['pass percentage'] = df['pass percentage'].str.replace('%', '').astype(float)

# Ensure 'obtained gpa - 5' is numeric
df['obtained gpa - 5'] = pd.to_numeric(df['obtained gpa - 5'])

# Prepare x (year) and y (metrics)
x = df['year'].astype(int)
y_pass = df['pass percentage']
y_gpa = df['obtained gpa - 5']

# Fit linear models
coeff_pass = np.polyfit(x, y_pass, 1)
coeff_gpa = np.polyfit(x, y_gpa, 1)

# Predict for 2011
year_2011 = 2011
predicted_pass = np.polyval(coeff_pass, year_2011)
predicted_gpa = np.polyval(coeff_gpa, year_2011)

print(f"Final Answer: {predicted_pass:.2f}, {predicted_gpa:.2f}")