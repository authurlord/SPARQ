import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('table.csv')

# Extract the 'Record high °F (°C)' row and parse Celsius values
record_high_celsius = df.loc[0, 'Jan':'Dec'].astype(str).str.extract(r'\(([^)]+)\)').apply(pd.to_numeric, errors='coerce')

# Ensure the month names are properly extracted and aligned
months = df.iloc[0, 1:13]  # First row, columns Jan to Dec
record_high_celsius = [float(x.strip('()')) if isinstance(x, str) else 0 for x in df.loc[0, 'Jan':'Dec'].astype(str).str.extract(r'\(([^)]+)\)').values]

# Create a bar chart
plt.figure(figsize=(12, 6))
plt.bar(months.index, record_high_celsius, color='skyblue')
plt.xlabel('Month')
plt.ylabel('Record High (°C)')
plt.title('Record High Temperatures in Celsius Degrees')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Print the values for verification
print(f"Final Answer: {record_high_celsius}")