import pandas as pd

df = pd.read_csv('table.csv')

# Calculate percentage change from 1971 to 2009
df['change_percent'] = ((df['number of bearers 2009'] - df['number of bearers 1971']) / df['number of bearers 1971']) * 100

# Identify significant deviations (absolute change > 20%)
deviations = df[df['change_percent'].abs() > 20]

# Extract the surnames that deviate significantly
deviant_surnames = deviations['surname'].tolist()

print(f"Final Answer: {', '.join(deviant_surnames)}")