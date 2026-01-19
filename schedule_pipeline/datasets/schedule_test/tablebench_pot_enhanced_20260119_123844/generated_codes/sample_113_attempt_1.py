import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Extract numeric value from 'Value' column (e.g., "1 cent" → 1)
df['Value'] = df['Value'].str.extract('(\d+)').astype(int)

# Convert 'Diameter' to numeric (remove ' mm' and convert)
df['Diameter'] = df['Diameter'].str.replace(' mm', '').astype(float)

# Create scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(df['Value'], df['Diameter'], color='blue')
plt.title('Relationship Between Coin Value and Diameter')
plt.xlabel('Coin Value (cents)')
plt.ylabel('Diameter (mm)')
plt.grid(True)
plt.show()