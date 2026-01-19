import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Convert 'pass percentage' to float by removing '%' and converting to float
df['pass percentage'] = df['pass percentage'].str.replace('%', '').astype(float)

# Extract year and target variables
years = df['year'].astype(int)
pass_percentages = df['pass percentage']
gpa_values = df['obtained gpa - 5']

# Fit linear models for both metrics
# For pass percentage
slope_pass, intercept_pass = np.polyfit(years, pass_percentages, 1)
predicted_pass_2011 = slope_pass * 2011 + intercept_pass

# For GPA
slope_gpa, intercept_gpa = np.polyfit(years, gpa_values, 1)
predicted_gpa_2011 = slope_gpa * 2011 + intercept_gpa

# Output the forecasted values
print(f"Final Answer: {predicted_pass_2011:.2f}, {predicted_gpa_2011:.2f}")