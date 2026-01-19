import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Clean and convert 'Value' to numeric (e.g., "1 cent" → 1)
df['Value'] = df['Value'].str.replace(' cents', '').str.replace(' dollar', '00').astype(int)

# Convert 'Diameter' to numeric (e.g., "18 mm" → 18)
df['Diameter'] = df['Diameter'].str.replace(' mm', '').astype(float)

# Create scatter plot
plt.figure(figsize=(8, 5))
plt.scatter(df['Value'], df['Diameter'], color='blue')
plt.title('Relationship Between Coin Value and Diameter')
plt.xlabel('Coin Value (in cents)')
plt.ylabel('Diameter (in mm)')
plt.grid(True)
plt.show()