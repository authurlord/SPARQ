import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Extract the 'Record high °F (°C)' row and parse Celsius values
record_high_celsius = []
for cell in df.loc[0, 'Jan':'Dec']:
    celsius_str = cell.split('(')[-1].rstrip(')')
    record_high_celsius.append(float(celsius_str))

# Months
months = df.columns[1:13]

# Plot bar chart
plt.figure(figsize=(10, 6))
plt.bar(months, record_high_celsius, color='skyblue')
plt.title('Record High Temperatures (°C) by Month')
plt.xlabel('Month')
plt.ylabel('Temperature (°C)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()