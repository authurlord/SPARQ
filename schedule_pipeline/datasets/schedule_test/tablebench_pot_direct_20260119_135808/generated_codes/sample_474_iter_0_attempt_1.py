import pandas as pd

df = pd.read_csv('table.csv')

# Check for anomalies
anomalies = []

# Check if total workers exceed sum of male and female
total_workers = int(df.loc[df['Particulars'] == 'Total Workers', 'Total'].values[0])
male_workers = int(df.loc[df['Particulars'] == 'Total Workers', 'Male'].values[0])
female_workers = int(df.loc[df['Particulars'] == 'Total Workers', 'Female'].values[0])
if male_workers + female_workers != total_workers:
    anomalies.append("Total Workers inconsistency: sum of male and female workers does not match total")

# Check for literacy percentages > 100%
male_literacy = df.loc[df['Particulars'] == 'Literacy', 'Male'].values[0]
if float(male_literacy.replace('%', '')) > 100:
    anomalies.append("Male literacy percentage exceeds 100%")

# Check if main workers are zero but total workers are non-zero
main_workers_total = int(df.loc[df['Particulars'] == 'Main Worker', 'Total'].values[0])
main_workers_male = int(df.loc[df['Particulars'] == 'Main Worker', 'Male'].values[0])
main_workers_female = int(df.loc[df['Particulars'] == 'Main Worker', 'Female'].values[0])
if main_workers_total != 0 and (main_workers_male != 0 or main_workers_female != 0):
    anomalies.append("Main Worker values are inconsistent with Total Workers")

# Print anomalies
for anomaly in anomalies:
    print(anomaly)

print(f"Final Answer: Male literacy percentage exceeds 100%, Total Workers inconsistency, Main Worker values inconsistent with Total Workers")