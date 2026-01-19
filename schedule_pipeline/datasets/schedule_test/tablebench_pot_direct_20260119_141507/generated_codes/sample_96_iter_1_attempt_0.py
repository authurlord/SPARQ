import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('table.csv')

# Extract the 'Record high °F (°C)' row and parse the Celsius values
record_high_celsius = df.loc[0, 'Jan':'Dec'].astype(str).str.split('\n').str[1].astype(float)

# Create a bar chart for record high temperatures in °C
plt.figure(figsize=(12, 6))
plt.bar(record_high_celsius.index, record_high_celsius, color='skyblue')
plt.title('Record High Temperatures in °C by Month')
plt.xlabel('Month')
plt.ylabel('Record High (°C)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Print the values for verification (optional)
print(f"Final Answer: {record_high_celsius.tolist()}")