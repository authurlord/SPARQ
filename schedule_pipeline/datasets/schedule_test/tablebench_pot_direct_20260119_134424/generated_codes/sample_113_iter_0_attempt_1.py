import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Extract and convert 'Value' to numeric
df['Value'] = df['Value'].str.replace(' cents', '').str.replace(' dollar', '').astype(int)

# Plot scatter plot
plt.scatter(df['Value'], df['Diameter'])
plt.xlabel('Coin Value (cents)')
plt.ylabel('Diameter (mm)')
plt.title('Relationship Between Coin Value and Diameter')
plt.grid(True)
plt.show()

print("Final Answer: Scatter plot generated.")