import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Convert 'Value' to numeric: extract numbers and convert to cents or dollars
def parse_value(value_str):
    if 'dollar' in value_str:
        return int(value_str.split()[0]) * 100
    else:
        return int(value_str.split()[0])

df['Value'] = df['Value'].apply(parse_value)
df['Diameter'] = df['Diameter'].str.replace(' mm', '').astype(float)

# Scatter plot
plt.figure(figsize=(8, 5))
plt.scatter(df['Value'], df['Diameter'], color='blue')
plt.title('Relationship Between Coin Value and Diameter')
plt.xlabel('Coin Value (in cents)')
plt.ylabel('Diameter (in mm)')
plt.grid(True)
plt.show()

print("Final Answer: Scatter plot generated.")