import pandas as pd

df = pd.read_csv('table.csv')

# Check for anomalies
anomalies = []

# Literacy rate > 100%
if df.loc[df['Particulars'] == 'Literacy', 'Male'].values[0] > '100 %':
    anomalies.append('Literacy rate for males exceeds 100%')

# Female workers > total population
female_workers = int(df.loc[df['Particulars'] == 'Total Workers', 'Female'].values[0])
total_population = int(df.loc[df['Particulars'] == 'Population', 'Total'].values[0])
if female_workers > total_population:
    anomalies.append('Female workers exceed total population')

# Female workers > total workers
total_workers = int(df.loc[df['Particulars'] == 'Total Workers', 'Total'].values[0])
if female_workers > total_workers:
    anomalies.append('Female workers exceed total workers')

# Print anomalies
print(f"Final Answer: {', '.join(anomalies)}")