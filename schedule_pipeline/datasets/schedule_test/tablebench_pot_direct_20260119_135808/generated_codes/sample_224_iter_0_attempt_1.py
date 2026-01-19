import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Convert 'p1 diameter (mm)' to float for plotting
df['p1 diameter (mm)'] = pd.to_numeric(df['p1 diameter (mm)'])

# Create a scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(df['p1 diameter (mm)'], df['p max ( bar )'], color='blue', label='Data Points')
plt.title('Maximum Pressure (p_max) vs Projectile Diameter (p1 diameter)')
plt.xlabel('Projectile Diameter (mm)')
plt.ylabel('Maximum Pressure (bar)')
plt.grid(True)
plt.legend()
plt.show()

# Since the question asks for how p_max changes with increasing p1 diameter,
# we can observe the trend from the plot. From the data, as diameter increases,
# p_max generally increases, though not strictly linearly.

# Final answer based on observed trend: p_max increases with increasing p1 diameter.
print("Final Answer: increases")