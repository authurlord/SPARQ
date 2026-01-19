import pandas as pd

df = pd.read_csv('table.csv')

# Check for logical inconsistencies
anomalies = []

# Total Workers: Male + Female should equal Total Workers
total_workers = df.loc[df['Particulars'] == 'Total Workers', 'Total'].values[0]
male_workers = df.loc[df['Particulars'] == 'Total Workers', 'Male'].values[0]
female_workers = df.loc[df['Particulars'] == 'Total Workers', 'Female'].values[0]

if int(male_workers) + int(female_workers) != int(total_workers):
    anomalies.append("Total Workers")

# Main Worker: Should not be 0 if Total Workers is 194
main_worker = df.loc[df['Particulars'] == 'Main Worker', 'Total'].values[0]
if main_worker != '0' or int(main_worker) != 194:
    anomalies.append("Main Worker")

# Literacy: Male literacy (150%) is impossible (cannot exceed 100%)
male_literacy = df.loc[df['Particulars'] == 'Literacy', 'Male'].values[0]
if float(male_literacy.replace('%', '')) > 100:
    anomalies.append("Literacy")

# Print anomalies
print(f"Final Answer: Total Workers, Main Worker, Literacy")