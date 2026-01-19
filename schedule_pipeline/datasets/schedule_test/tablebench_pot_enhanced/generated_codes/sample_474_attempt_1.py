import pandas as pd

df = pd.read_csv('table.csv')

# Check for anomalies
anomalies = []

# Literacy rate for males is 150%, which is impossible
if float(df.loc[df['Particulars'] == 'Literacy', 'Male'].values[0].replace('%', '')) > 100:
    anomalies.append('Literacy (Male)')

# Female workers (1000) exceed total workers (194)
if int(df.loc[df['Particulars'] == 'Total Workers', 'Female'].values[0]) > int(df.loc[df['Particulars'] == 'Total Workers', 'Total'].values[0]):
    anomalies.append('Total Workers (Female)')

# Main Worker has 0 males and 0 females, but Total Workers is 194
if int(df.loc[df['Particulars'] == 'Main Worker', 'Male'].values[0]) == 0 and int(df.loc[df['Particulars'] == 'Main Worker', 'Female'].values[0]) == 0:
    anomalies.append('Main Worker')

# Print anomalies
print(f"Final Answer: {', '.join(anomalies)}")