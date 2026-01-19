import pandas as pd

df = pd.read_csv('table.csv')
# Filter the row where 'language' is 'spanish'
spanish_row = df[df['language'] == 'spanish']
# Extract the values for each municipality
municipalities = spanish_row.columns[1:]  # Exclude the 'language' column
# Count how many municipalities have population >= 40,000
count = sum(int(spanish_row[col].iloc[0]) >= 40000 for col in municipalities)
print(f"Final Answer: {count}")