import pandas as pd
import matplotlib.pyplot as plt

# Load the table
df = pd.read_csv('table.csv')

# Extract relevant data
quarters = ['Quarter Ended March 31', 'Quarter Ended June 30', 'Quarter Ended September 30', 'Quarter Ended December 31']

# Clean and convert operating revenue values
revenue_2019 = [float(row[1].replace('$', '')) for row in df[df['Row Header'] == 'Operating revenue'].values]
revenue_2018 = [float(row[1].replace('$', '')) for row in df[df['Row Header'] == '2018 Operating revenue'].values]

# Create a DataFrame for plotting
data = {
    '2019 Operating Revenue': revenue_2019,
    '2018 Operating Revenue': revenue_2018
}
df_plot = pd.DataFrame(data, index=quarters)

# Plot the line chart
plt.figure(figsize=(10, 6))
plt.plot(df_plot.index, df_plot['2019 Operating Revenue'], marker='o', label='2019 Operating Revenue')
plt.plot(df_plot.index, df_plot['2018 Operating Revenue'], marker='s', label='2018 Operating Revenue')
plt.title('Company Operating Revenue by Quarter')
plt.xlabel('Quarter')
plt.ylabel('Operating Revenue ($)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()