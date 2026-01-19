import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Extract numeric value from the 'Value' column (e.g., "1 cent" -> 1)
df['Value'] = df['Value'].str.replace(' cent', '').str.replace(' dollars', '').str.strip()
df['Value'] = pd.to_numeric(df['Value'], errors='coerce')

# Convert Diameter to numeric
df['Diameter'] = pd.to_numeric(df['Diameter'], errors='coerce')

# Drop rows with missing values
df = df.dropna(subset=['Value', 'Diameter'])

# Create scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(df['Value'], df['Diameter'], color='blue')
plt.title('Relationship between Coin Value and Diameter')
plt.xlabel('Coin Value')
plt.ylabel('Diameter (mm)')
plt.grid(True)
plt.show()

print(f"Final Answer: Scatter plot generated successfully")