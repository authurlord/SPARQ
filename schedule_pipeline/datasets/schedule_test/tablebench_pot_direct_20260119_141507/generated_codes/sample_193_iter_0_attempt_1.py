import pandas as pd

df = pd.read_csv('table.csv')

# Clean the '% renewable of total electricity demand' column by removing '%' and converting to float
df['% renewable of total electricity demand'] = df['% renewable of total electricity demand'].str.replace('%', '').astype(float)

# Extract wind power and % renewable columns
wind_power = df['wind power']
renewable_percent = df['% renewable of total electricity demand']

# Calculate the correlation
correlation = wind_power.corr(renewable_percent)

print(f"Final Answer: {correlation:.3f}")