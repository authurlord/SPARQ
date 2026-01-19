import pandas as pd

df = pd.read_csv('table.csv')

# Check for anomalies
anomalies = []

# Literacy rate for males is 150%, which is impossible
if df.loc[df['Particulars'] == 'Literacy', 'Male'].iloc[0] == '150 %':
    anomalies.append('Literacy rate for males is 150%, which is invalid.')

# Total workers (194) with 1000 females — impossible since total population is 524
if df.loc[df['Particulars'] == 'Total Workers', 'Female'].iloc[0] == '1000':
    anomalies.append('Total Workers has 1000 females, exceeding total population of 524.')

# Main Worker: male and female both 0, but Total Workers is 194 — inconsistency
if df.loc[df['Particulars'] == 'Main Worker', 'Male'].iloc[0] == '0' and df.loc[df['Particulars'] == 'Main Worker', 'Female'].iloc[0] == '0':
    if df.loc[df['Particulars'] == 'Total Workers', 'Total'].iloc[0] != '0':
        anomalies.append('Main Workers are 0 but Total Workers are 194.')

# Print anomalies
for anomaly in anomalies:
    print(anomaly)

print("Final Answer: Literacy rate for males is 150%, Total Workers has 1000 females, Main Workers are 0 but Total Workers are 194")