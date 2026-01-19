import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('table.csv')

# Extract relevant columns and convert to numeric
quarters = ['Quarter Ended March 31', 'Quarter Ended June 30', 'Quarter Ended September 30', 'Quarter Ended December 31']
operating_revenue = df[df['Row Header'] == 'Operating revenue'][quarters].iloc[0].str.replace('$', '').astype(float)
twenty_eight_operating_revenue = df[df['Row Header'] == '2018 Operating revenue'][quarters].iloc[0].str.replace('$', '').astype(float)

# Create a DataFrame for plotting
data = pd.DataFrame({
    'Operating Revenue': operating_revenue,
    '2018 Operating Revenue': twenty_eight_operating_revenue
}, index=quarters)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(data.index, data['Operating Revenue'], marker='o', label='Operating Revenue')
plt.plot(data.index, data['2018 Operating Revenue'], marker='s', label='2018 Operating Revenue')
plt.title('Company Operating Revenue by Quarter')
plt.xlabel('Quarter')
plt.ylabel('Operating Revenue ($)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Final Answer: The chart has been created and displayed
Final Answer: chart_created