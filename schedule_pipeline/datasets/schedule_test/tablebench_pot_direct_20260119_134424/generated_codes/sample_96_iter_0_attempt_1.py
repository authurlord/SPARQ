import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Extract the row with record high temperatures
record_high_row = df[df['Month'] == 'Record high °F (°C)'].iloc[0]

# Parse Celsius values from the string (e.g., "76\n(24)" -> 24)
celsius_values = [int(entry.split('(')[-1].rstrip(')')) for entry in record_high_row[1:-1]]  # Exclude 'Year' column

# Months list
months = df.columns[1:-1]  # Exclude 'Month' and 'Year'

# Plot bar chart
plt.figure(figsize=(10, 5))
plt.bar(months, celsius_values, color='red')
plt.title('Record High Temperatures (°C) by Month')
plt.xlabel('Month')
plt.ylabel('Temperature (°C)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print(f"Final Answer: Bar chart displayed")