import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Convert 'Value' to numeric (e.g., "1 cent" -> 1, "2 dollars" -> 200)
def parse_value(value_str):
    if 'cent' in value_str:
        return int(value_str.split()[0])
    elif 'dollar' in value_str:
        return int(value_str.split()[0]) * 100
    return 0

df['Value_numeric'] = df['Value'].apply(parse_value)

# Convert 'Diameter' to float
df['Diameter'] = df['Diameter'].str.replace(' mm', '').astype(float)

# Create scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(df['Value_numeric'], df['Diameter'], color='blue')
plt.title('Relationship Between Coin Value and Diameter')
plt.xlabel('Coin Value (in cents)')
plt.ylabel('Diameter (in mm)')
plt.grid(True)
plt.show()