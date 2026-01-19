import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Parse the 'Value' column to extract numeric value (e.g., "1 cent" -> 1)
df['Value_num'] = df['Value'].str.extract(r'(\d+)').astype(float)

# Extract diameter
diameters = df['Diameter'].astype(float)

# Create scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(df['Value_num'], diameters, color='blue')
plt.title('Relationship between Coin Value and Diameter')
plt.xlabel('Coin Value')
plt.ylabel('Diameter (mm)')
plt.grid(True)
plt.show()