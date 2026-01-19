import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('table.csv')

# Extract the relevant columns
quarters = ['Quarter Ended March 31', 'Quarter Ended June 30', 'Quarter Ended September 30', 'Quarter Ended December 31']
revenue_2018 = df.loc[df['Row Header'] == '2018 Operating revenue', quarters].iloc[0].str.replace('$', '').astype(float)
revenue_other = df.loc[df['Row Header'] == 'Operating revenue', quarters].astype(str).str.replace('$', '').astype(float)

# Create the line chart
plt.figure(figsize=(10, 6))
plt.plot(quarters, revenue_other, marker='o', label='Operating Revenue (Non-2018)', color='blue')
plt.plot(quarters, revenue_2018, marker='s', label='2018 Operating Revenue', color='red')

plt.title('Operating Revenue Trend by Quarter')
plt.xlabel('Quarter Ended')
plt.ylabel('Operating Revenue ($)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()