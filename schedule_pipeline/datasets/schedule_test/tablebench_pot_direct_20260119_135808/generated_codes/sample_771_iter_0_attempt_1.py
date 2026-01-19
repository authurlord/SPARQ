import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Convert 'pass percentage' to float by removing '%' and converting
df['pass percentage'] = df['pass percentage'].str.replace('%', '').astype(float)

# Extract year and metrics
years = df['year'].astype(int)
pass_percentages = df['pass percentage']
gpa_5 = df['obtained gpa - 5']

# Calculate average yearly increase for pass percentage
slope_pass = np.polyfit(years, pass_percentages, 1)[0]
forecast_pass = pass_percentages[-1] + slope_pass

# Calculate average yearly increase for GPA
slope_gpa = np.polyfit(years, gpa_5, 1)[0]
forecast_gpa = gpa_5[-1] + slope_gpa

print(f"Final Answer: {forecast_pass:.2f}, {forecast_gpa:.2f}")