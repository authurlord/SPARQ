import pandas as pd

df = pd.read_csv('table.csv')

# Convert 'pass percentage' to numeric by removing '%' and converting to float
df['pass percentage'] = df['pass percentage'].str.replace('%', '').astype(float)

# Extract year and metrics
years = df['year'].astype(int)
pass_percentages = df['pass percentage']
gpa_values = df['obtained gpa - 5']

# Calculate average annual increase
annual_increase_pass = (pass_percentages.iloc[-1] - pass_percentages.iloc[0]) / (len(years) - 1)
annual_increase_gpa = (gpa_values.iloc[-1] - gpa_values.iloc[0]) / (len(years) - 1)

# Forecast for 2011
forecast_pass = pass_percentages.iloc[-1] + annual_increase_pass
forecast_gpa = gpa_values.iloc[-1] + annual_increase_gpa

print(f"Final Answer: {forecast_pass:.2f}, {forecast_gpa:.2f}")