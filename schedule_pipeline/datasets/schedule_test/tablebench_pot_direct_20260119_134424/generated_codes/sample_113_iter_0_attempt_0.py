import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Extract and clean data
df['Value'] = df['Value'].str.replace(' cents', '').astype(int)
df['Diameter'] = df['Diameter'].str.replace(' mm', '').astype(float)

# Create scatter plot
plt.figure(figsize=(8, 5))
plt.scatter(df['Value'], df['Diameter'], color='blue')
plt.title('Relationship Between Coin Value and Diameter')
plt.xlabel('Coin Value (cents)')
plt.ylabel('Diameter (mm)')
plt.grid(True)
plt.show()

print("Final Answer: Scatter plot generated.")