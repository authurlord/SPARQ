import pandas as pd

df = pd.read_csv('table.csv')

# Identify and print rows with suspicious or impossible values
anomalies = []

# Check literacy rate for males (150% is invalid)
if df.loc[df['Particulars'] == 'Literacy', 'Male'].values[0] == '150 %':
    anomalies.append("Literacy (Male): 150% is impossible — likely a typo or data entry error.")

# Check total workers: female value is 1000, but total is 194
if df.loc[df['Particulars'] == 'Total Workers', 'Female'].values[0] == '1000':
    anomalies.append("Total Workers (Female): 1000 contradicts total of 194 — clearly an error.")

# Check if male + female population equals total population
population_row = df[df['Particulars'] == 'Population']
if not population_row.empty:
    total_pop = int(population_row['Total'].values[0])
    male_pop = int(population_row['Male'].values[0]) if population_row['Male'].values[0] != '-' else 0
    female_pop = int(population_row['Female'].values[0]) if population_row['Female'].values[0] != '-' else 0
    if male_pop + female_pop != total_pop:
        anomalies.append(f"Population: Male ({male_pop}) + Female ({female_pop}) = {male_pop + female_pop}, but total is {total_pop} — mismatch.")

# Check child count: 46 + 49 = 95 → matches
child_row = df[df['Particulars'] == 'Child (0-6)']
if not child_row.empty:
    child_total = int(child_row['Total'].values[0])
    child_male = int(child_row['Male'].values[0]) if child_row['Male'].values[0] != '-' else 0
    child_female = int(child_row['Female'].values[0]) if child_row['Female'].values[0] != '-' else 0
    if child_male + child_female != child_total:
        anomalies.append(f"Child (0-6): Male ({child_male}) + Female ({child_female}) = {child_male + child_female}, but total is {child_total} — mismatch.")

# Check Schedule Caste: 137 + 138 = 275 → matches
caste_row = df[df['Particulars'] == 'Schedule Caste']
if not caste_row.empty:
    caste_total = int(caste_row['Total'].values[0])
    caste_male = int(caste_row['Male'].values[0]) if caste_row['Male'].values[0] != '-' else 0
    caste_female = int(caste_row['Female'].values[0]) if caste_row['Female'].values[0] != '-' else 0
    if caste_male + caste_female != caste_total:
        anomalies.append(f"Schedule Caste: Male ({caste_male}) + Female ({caste_female}) = {caste_male + caste_female}, but total is {caste_total} — mismatch.")

# Print the identified anomalies
if anomalies:
    print("Anomalies found:")
    for anomaly in anomalies:
        print(anomaly)
else:
    print("No significant deviations found.")

Final Answer: Literacy (Male), Total Workers (Female)