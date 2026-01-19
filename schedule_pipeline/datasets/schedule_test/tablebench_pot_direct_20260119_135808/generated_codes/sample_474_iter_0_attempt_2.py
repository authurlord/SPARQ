import pandas as pd

df = pd.read_csv('table.csv')

# Check for anomalies
anomalies = []

# Literacy: Male literacy is 150%, which is impossible
if df.loc[df['Particulars'] == 'Literacy', 'Male'].iloc[0] == '150 %':
    anomalies.append('Literacy - Male literacy rate exceeds 100%')

# Total Workers: Female count (1000) exceeds total population (524)
if int(df.loc[df['Particulars'] == 'Total Workers', 'Female'].iloc[0]) > int(df.loc[df['Particulars'] == 'Population', 'Total'].iloc[0]):
    anomalies.append('Total Workers - Female workers exceed total population')

# Main Worker: Both male and female are 0, but total workers are 194
if int(df.loc[df['Particulars'] == 'Main Worker', 'Male'].iloc[0]) == 0 and int(df.loc[df['Particulars'] == 'Main Worker', 'Female'].iloc[0]) == 0:
    if int(df.loc[df['Particulars'] == 'Total Workers', 'Total'].iloc[0]) > 0:
        anomalies.append('Main Worker - Zero main workers despite non-zero total workers')

# Print anomalies
print(f"Final Answer: {', '.join(anomalies)}")