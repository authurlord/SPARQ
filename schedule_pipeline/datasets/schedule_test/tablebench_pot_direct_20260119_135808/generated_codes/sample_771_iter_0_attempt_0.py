import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Convert 'pass percentage' to float by removing the % sign
df['pass percentage'] = df['pass percentage'].str.replace('%', '').astype(float)

# Convert 'obtained gpa - 5' to numeric
df['obtained gpa - 5'] = pd.to_numeric(df['obtained gpa - 5'])

# Extract year and target variables
years = df['year'].astype(int)
pass_percentage = df['pass percentage']
gpa = df['obtained gpa - 5']

# Fit linear models for both metrics
slope_pass, intercept_pass = np.polyfit(years, pass_percentage, 1)
slope_gpa, intercept_gpa = np.polyfit(years, gpa, 1)

# Predict for 2011
predicted_pass_percentage = slope_pass * 2011 + intercept_pass
predicted_gpa = slope_gpa * 2011 + intercept_gpa

print(f"Final Answer: {predicted_pass_percentage:.2f}, {predicted_gpa:.2f}")