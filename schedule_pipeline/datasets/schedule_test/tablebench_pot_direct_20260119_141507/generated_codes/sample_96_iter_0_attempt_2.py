import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('table.csv')

# Extract the 'Record high °F (°C)' row and parse the Celsius values
record_high_celsius = df.iloc[0][1:].astype(str).str.extract(r'\(([^)]+)\)')[0].astype(float)

# Create a list of months for the x-axis
months = df.iloc[0][0:12]  # First row, excluding 'Record high °F (°C)' header

# Create a bar chart
plt.figure(figsize=(12, 6))
plt.bar(months, record_high_celsius, color='skyblue')
plt.title('Record High Temperatures in Celsius Degrees')
plt.xlabel('Month')
plt.ylabel('Record High (°C)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Print the values for verification (optional)
print(f"Final Answer: {record_high_celsius.tolist()}")