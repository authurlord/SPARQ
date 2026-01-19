import pandas as pd
import matplotlib.pyplot as plt

# Load the table
df = pd.read_csv('table.csv')

# Extract the 'Record high °F (°C)' row and parse Celsius values
record_high_celsius = []
for cell in df.loc[0, 'Jan':].dropna().values:
    # Extract the value in parentheses
    celsius_val = cell.split('\n')[-1] if '\n' in cell else cell
    try:
        celsius_val = float(celsius_val.strip('()'))
        record_high_celsius.append(celsius_val)
    except:
        record_high_celsius.append(0)

# Create a list of months (excluding the first row which is headers)
months = df.columns[1:-1]  # Exclude 'Year'

# Create a bar chart
plt.figure(figsize=(12, 6))
plt.bar(months, record_high_celsius, color='skyblue')
plt.xlabel('Month')
plt.ylabel('Record High (°C)')
plt.title('Record High Temperatures in Celsius by Month')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Print the extracted Celsius values for verification (optional)
print(f"Final Answer: {record_high_celsius}")