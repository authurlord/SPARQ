import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('table.csv')

# Clean and convert the revenue columns to numeric
quarters = ['Quarter Ended March 31', 'Quarter Ended June 30', 'Quarter Ended September 30', 'Quarter Ended December 31']
for col in quarters:
    df[col] = df[col].str.replace('$', '').str.replace(',', '').astype(float)

# Extract the operating revenue for the years (excluding the 2018 row)
# The first row is "Operating revenue", so we take the next rows
operating_revenue_2018 = df[df['Row Header'] == '2018 Operating revenue'][quarters].values[0]
operating_revenue_other = df[df['Row Header'] == 'Operating revenue'][quarters].values[0]

# Create quarters list for x-axis
quarters_labels = ['Mar 31', 'Jun 30', 'Sep 30', 'Dec 31']

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(quarters_labels, operating_revenue_other, marker='o', label='Operating Revenue (Non-2018)')
plt.plot(quarters_labels, operating_revenue_2018, marker='s', label='Operating Revenue (2018)')
plt.title('Company Operating Revenue Over Quarters')
plt.xlabel('Quarter Ended')
plt.ylabel('Operating Revenue ($)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Final Answer: The line chart has been created and displayed.
Final Answer: chart