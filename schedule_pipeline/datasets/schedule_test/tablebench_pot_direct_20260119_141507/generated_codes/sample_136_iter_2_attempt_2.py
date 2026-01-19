import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('table.csv')

# Clean the revenue columns by removing $ and commas, then convert to float
quarters = ['Quarter Ended March 31', 'Quarter Ended June 30', 'Quarter Ended September 30', 'Quarter Ended December 31']
df['Operating revenue'] = df['Operating revenue'].str.replace('$', '').str.replace(',', '').astype(float)
df['2018 Operating revenue'] = df['2018 Operating revenue'].str.replace('$', '').str.replace(',', '').astype(float)

# Extract the quarterly values for plotting
quarters_values = df[quarters]
revenue_2018 = df['2018 Operating revenue']

# Create the line chart
plt.figure(figsize=(10, 6))
plt.plot(quarters, df['Operating revenue'], marker='o', label='Non-2018 Operating Revenue')
plt.plot(quarters, revenue_2018, marker='s', label='2018 Operating Revenue')
plt.title('Company Operating Revenue Over Quarters (with 2018 Comparison)')
plt.xlabel('Quarter Ended')
plt.ylabel('Operating Revenue ($)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()