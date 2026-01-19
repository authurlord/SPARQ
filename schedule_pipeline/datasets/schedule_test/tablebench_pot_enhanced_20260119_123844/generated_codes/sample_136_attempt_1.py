import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('table.csv')

# Extract the operating revenue data for both years
operating_revenue = df[df['Row Header'] == 'Operating revenue']
revenue_2018 = df[df['Row Header'] == '2018 Operating revenue']

# Clean and convert values to integers (remove $ and convert to int)
quarters = ['Quarter Ended March 31', 'Quarter Ended June 30', 'Quarter Ended September 30', 'Quarter Ended December 31']
current_year_revenue = [int(x.replace('$', '')) for x in operating_revenue.iloc[0][1:]]
past_year_revenue = [int(x.replace('$', '')) for x in revenue_2018.iloc[0][1:]]

# Plot the line chart
plt.figure(figsize=(10, 6))
plt.plot(quarters, current_year_revenue, marker='o', label='Current Year')
plt.plot(quarters, past_year_revenue, marker='s', label='2018')

# Add labels and title
plt.title('Operating Revenue Over Quarters (Current Year vs 2018)')
plt.xlabel('Quarter')
plt.ylabel('Operating Revenue ($)')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()

# Show the plot
plt.show()