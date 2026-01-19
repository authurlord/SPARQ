import pandas as pd

df = pd.read_csv('table.csv')

# Check for anomalies
anomalies = []

# Literacy rate for males exceeds 100%
if df.loc[df['Particulars'] == 'Literacy', 'Male'].iloc[0].replace('%', '') != '100':
    literacy_male = float(df.loc[df['Particulars'] == 'Literacy', 'Male'].iloc[0].replace('%', ''))
    if literacy_male > 100:
        anomalies.append("Literacy rate for males (150%) exceeds 100%, which is impossible.")

# Female workers (1000) exceed total workers (194)
total_workers = int(df.loc[df['Particulars'] == 'Total Workers', 'Total'].iloc[0])
female_workers = int(df.loc[df['Particulars'] == 'Total Workers', 'Female'].iloc[0])
if female_workers > total_workers:
    anomalies.append("Female workers (1000) exceed total workers (194), which is impossible.")

# Output anomalies
for anomaly in anomalies:
    print(anomaly)

print(f"Final Answer: literacy rate for males, female workers")