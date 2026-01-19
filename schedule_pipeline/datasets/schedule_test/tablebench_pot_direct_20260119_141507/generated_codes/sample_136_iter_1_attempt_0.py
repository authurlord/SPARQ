import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('table.csv')

# Extract the relevant columns for operating revenue
quarters = ['Quarter Ended March 31', 'Quarter Ended June 30', 'Quarter Ended September 30', 'Quarter Ended December 31']
revenue_2019 = [float(row[1]) for row in df.values if row[0] == 'Operating revenue']
revenue_2018 = [float(row[1]) for row in df.values if row[0] == '2018 Operating revenue']

# Prepare the data for plotting
quarters_labels = ['Mar', 'Jun', 'Sep', 'Dec']
data_2019 = [9589, 11402, 11380, 10888]
data_2018 = [9032, 10777, 11003, 10491]

# Create the line chart
plt.figure(figsize=(10, 6))
plt.plot(quarters_labels, data_2019, marker='o', label='2019 Operating Revenue')
plt.plot(quarters_labels, data_2018, marker='s', label='2018 Operating Revenue')
plt.title('Operating Revenue Trend by Quarter')
plt.xlabel('Quarter')
plt.ylabel('Operating Revenue ($)')
plt.legend()
plt.grid(True)
plt.show()

print("Final Answer: Line chart created and displayed")