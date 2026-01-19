import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('table.csv')

# Extract the record high temperatures in Celsius
record_high_celsius = []
for cell in df.loc[0, 'Jan':'Dec']:
    # Extract the value inside parentheses (Celsius)
    celsius_str = cell.split('(')[-1].strip(')')
    record_high_celsius.append(float(celsius_str))

# Months
months = df.columns[1:13]  # Jan to Dec

# Plot bar chart
plt.figure(figsize=(10, 6))
plt.bar(months, record_high_celsius, color='red')
plt.title('Record High Temperatures (°C) by Month')
plt.xlabel('Month')
plt.ylabel('Temperature (°C)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Print final answer (the data used for the chart)
print(f"Final Answer: {', '.join([str(int(x)) for x in record_high_celsius])}")