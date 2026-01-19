import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Convert columns to numeric
df['p1 diameter (mm)'] = pd.to_numeric(df['p1 diameter (mm)'])
df['p max ( bar )'] = pd.to_numeric(df['p max ( bar )'])

# Create scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(df['p1 diameter (mm)'], df['p max ( bar )'], color='blue', label='Data Points')

# Add trend line
z = np.polyfit(df['p1 diameter (mm)'], df['p max ( bar )'], 1)
p = np.poly1d(z)
plt.plot(df['p1 diameter (mm)'], p(df['p1 diameter (mm)']), color='red', linestyle='--', label='Trend Line')

plt.title('Maximum Pressure (p max) vs Projectile Diameter (p1 diameter)')
plt.xlabel('Projectile Diameter (mm)')
plt.ylabel('Maximum Pressure (bar)')
plt.legend()
plt.grid(True)
plt.show()

# Final Answer: Based on the plot, p max generally increases with increasing p1 diameter, though not strictly linear.
print("Final Answer: increases")