import pandas as pd

df = pd.read_csv('table.csv')
# Extract wind power and % renewable of total electricity demand
wind_power = df['wind power']
renewable_percentage = df['% renewable of total electricity demand']

# Calculate the correlation
correlation = wind_power.corr(renewable_percentage)
print(f"Final Answer: {correlation:.3f}")