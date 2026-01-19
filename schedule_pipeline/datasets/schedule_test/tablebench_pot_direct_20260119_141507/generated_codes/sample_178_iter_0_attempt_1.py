import pandas as pd

df = pd.read_csv('table.csv')
# Extract power and max speed columns
power_hp = df['power (horsepower)']
max_speed = df['max speed (km / h)']

# Calculate the correlation between power and max speed
correlation = power_hp.corr(max_speed)
print(f"Final Answer: {correlation:.2f}")