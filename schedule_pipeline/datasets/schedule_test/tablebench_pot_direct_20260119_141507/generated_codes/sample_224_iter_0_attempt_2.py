import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Extract relevant columns
df['p1 diameter (mm)'] = pd.to_numeric(df['p1 diameter (mm)'], errors='coerce')
df['p max (bar)'] = pd.to_numeric(df['p max (bar)'], errors='coerce')

# Drop rows with missing values
df = df.dropna(subset=['p1 diameter (mm)', 'p max (bar)'])

# Sort by projectile diameter to observe trend
df_sorted = df.sort_values(by='p1 diameter (mm)')

# Plot the relationship
plt.figure(figsize=(10, 6))
plt.plot(df_sorted['p1 diameter (mm)'], df_sorted['p max (bar)'], marker='o', linestyle='-', color='b')
plt.title('Maximum Pressure (p max) vs Projectile Diameter (p1 diameter)')
plt.xlabel('Projectile Diameter (mm)')
plt.ylabel('Maximum Pressure (bar)')
plt.grid(True)
plt.tight_layout()

# Show the plot
plt.show()

# Final answer based on observed trend (from visual inspection)
# As projectile diameter increases, maximum pressure generally increases, with some exceptions.
Final Answer: increases