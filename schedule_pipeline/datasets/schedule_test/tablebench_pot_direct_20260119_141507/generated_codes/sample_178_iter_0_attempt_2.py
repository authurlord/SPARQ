import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Extract relevant columns
power_hp = df['power (horsepower)'].astype(float)
max_speed = df['max speed (km / h)'].astype(float)

# Calculate correlation
correlation = power_hp.corr(max_speed)

# Create scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(power_hp, max_speed, alpha=0.7)
plt.title('Relationship between Power (Horsepower) and Maximum Speed (km/h)')
plt.xlabel('Power (Horsepower)')
plt.ylabel('Maximum Speed (km/h)')
plt.grid(True)
plt.show()

print(f"Final Answer: {correlation:.2f}")