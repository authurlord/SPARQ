import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Convert 'Value' to numeric: extract number from string (e.g., "1 cent" → 1, "2 dollars" → 200)
df['Value'] = df['Value'].str.extract('(\d+)').astype(int)
df['Value'] = df['Value'].where(df['Value'] < 100, df['Value'] * 100)  # Convert dollars to cents

# Convert 'Diameter' to float
df['Diameter'] = df['Diameter'].str.extract('(\d+\.\d+|\d+)').astype(float)

# Create scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(df['Value'], df['Diameter'], color='blue')
plt.title('Relationship Between Coin Value and Diameter')
plt.xlabel('Coin Value (in cents)')
plt.ylabel('Diameter (mm)')
plt.grid(True)
plt.show()

print("Final Answer: Scatter plot generated.")