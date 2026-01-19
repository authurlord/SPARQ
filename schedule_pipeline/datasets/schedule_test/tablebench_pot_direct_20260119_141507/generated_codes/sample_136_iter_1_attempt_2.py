import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('table.csv')

# Extract the quarters (as index)
quarters = ['Quarter Ended March 31', 'Quarter Ended June 30', 'Quarter Ended September 30', 'Quarter Ended December 31']

# Extract the operating revenue values
operating_revenue = df[df['Row Header'] == 'Operating revenue'][quarters].iloc[0].astype(float)
twenty_eight_operating_revenue = df[df['Row Header'] == '2018 Operating revenue'][quarters].iloc[0].astype(float)

# Create a DataFrame for plotting
data = pd.DataFrame({
    'Operating Revenue': operating_revenue,
    '2018 Operating Revenue': twenty_eight_operating_revenue
}, index=quarters)

# Plot the line chart
plt.figure(figsize=(10, 6))
plt.plot(data.index, data['Operating Revenue'], marker='o', label='Operating Revenue (Non-2018)', color='blue')
plt.plot(data.index, data['2018 Operating Revenue'], marker='s', label='2018 Operating Revenue', color='red')
plt.title('Company Operating Revenue Over Quarters')
plt.xlabel('Quarter')
plt.ylabel('Operating Revenue ($)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Final Answer: The chart is generated and displayed.
Final Answer: chart_generated