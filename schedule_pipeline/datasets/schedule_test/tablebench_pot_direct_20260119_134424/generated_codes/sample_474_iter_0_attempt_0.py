import pandas as pd

df = pd.read_csv('table.csv')

# Check for anomalies
anomalies = []

# Literacy rate > 100%
if df.loc[df['Particulars'] == 'Literacy', 'Male'].values[0].strip('%') > '100':
    anomalies.append("Male literacy rate exceeds 100%")

# Female workers > total workers
total_workers = int(df.loc[df['Particulars'] == 'Total Workers', 'Total'].values[0])
female_workers = int(df.loc[df['Particulars'] == 'Total Workers', 'Female'].values[0])
if female_workers > total_workers:
    anomalies.append("Female workers exceed total workers")

# Female population < female workers
female_population = int(df.loc[df['Particulars'] == 'Population', 'Female'].values[0])
if female_workers > female_population:
    anomalies.append("Female workers exceed female population")

# Print anomalies
print(f"Final Answer: {', '.join(anomalies)}")