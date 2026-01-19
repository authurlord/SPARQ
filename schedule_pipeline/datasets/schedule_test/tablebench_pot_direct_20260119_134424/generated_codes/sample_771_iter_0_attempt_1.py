import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Convert 'pass percentage' to float by removing '%'
df['pass percentage'] = df['pass percentage'].str.replace('%', '').astype(float)

# Extract year and metrics
years = df['year'].astype(int)
pass_percentages = df['pass percentage']
gpa_values = df['obtained gpa - 5']

# Fit linear models for both metrics
poly_pass = np.polyfit(years, pass_percentages, 1)
poly_gpa = np.polyfit(years, gpa_values, 1)

# Predict for 2011
predicted_pass = np.polyval(poly_pass, 2011)
predicted_gpa = np.polyval(poly_gpa, 2011)

print(f"Final Answer: {predicted_pass:.2f}, {predicted_gpa:.2f}")