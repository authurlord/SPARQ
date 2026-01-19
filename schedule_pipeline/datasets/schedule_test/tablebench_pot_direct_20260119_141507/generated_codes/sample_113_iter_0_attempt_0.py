import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Map value to numeric: extract number from strings like "1 cent", "2 cents"
df['Value_numeric'] = df['Value'].str.extract(r'(\d+)').astype(float)

# Create scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(df['Value_numeric'], df['Diameter'], color='blue')
plt.title('Relationship between Coin Value and Diameter')
plt.xlabel('Coin Value (in cents)')
plt.ylabel('Diameter (mm)')
plt.grid(True)
plt.show()