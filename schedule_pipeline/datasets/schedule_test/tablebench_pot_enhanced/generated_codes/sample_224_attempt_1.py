import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Convert columns to numeric for plotting
df['p1 diameter (mm)'] = pd.to_numeric(df['p1 diameter (mm)'])
df['p max ( bar )'] = pd.to_numeric(df['p max ( bar )'])

# Create scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(df['p1 diameter (mm)'], df['p max ( bar )'], color='blue')
plt.title('Maximum Pressure (p_max) vs Projectile Diameter (p1 diameter)')
plt.xlabel('Projectile Diameter (mm)')
plt.ylabel('Maximum Pressure (bar)')
plt.grid(True)
plt.show()

# Print the trend observation
print("Final Answer: Increasing")