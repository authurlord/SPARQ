import pandas as pd

df = pd.read_csv('table.csv')

# Check for anomalies
anomalies = []

# Literacy rate for males is 150% — impossible
if float(df.loc[df['Particulars'] == 'Literacy', 'Male'].values[0].strip('%')) > 100:
    anomalies.append('Literacy (Male)')

# Total Workers = 194, but Female = 1000 — impossible
if int(df.loc[df['Particulars'] == 'Total Workers', 'Female'].values[0]) > int(df.loc[df['Particulars'] == 'Total Workers', 'Total'].values[0]):
    anomalies.append('Total Workers (Female)')

# Main Worker = 194, but both Male and Female are 0 — contradiction
if int(df.loc[df['Particulars'] == 'Main Worker', 'Male'].values[0]) == 0 and int(df.loc[df['Particulars'] == 'Main Worker', 'Female'].values[0]) == 0:
    anomalies.append('Main Worker')

# Print anomalies
print(f"Final Answer: {', '.join(anomalies)}")