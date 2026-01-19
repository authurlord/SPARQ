import pandas as pd

# Load the dataframe
df = pd.read_csv('table.csv')

# Convert year column to integer for calculations
df['year'] = pd.to_numeric(df['year'])

# Extract the relevant columns
pass_percentage = df['pass percentage'].str.replace('%', '').astype(float)
gpa = df['obtained gpa - 5'].astype(int)

# Calculate the linear trend for pass percentage and GPA from 2005 to 2010
years = df['year']
pass_trend = (pass_percentage.iloc[-1] - pass_percentage.iloc[0]) / (years.iloc[-1] - years.iloc[0])
gpa_trend = (gpa.iloc[-1] - gpa.iloc[0]) / (years.iloc[-1] - years.iloc[0])

# Project to 2011
predicted_pass_percentage = pass_percentage.iloc[0] + pass_trend * (2011 - years.iloc[0])
predicted_gpa = gpa.iloc[0] + gpa_trend * (2011 - years.iloc[0])

print(f"Final Answer: {predicted_pass_percentage:.2f}%, {predicted_gpa:.0f}")