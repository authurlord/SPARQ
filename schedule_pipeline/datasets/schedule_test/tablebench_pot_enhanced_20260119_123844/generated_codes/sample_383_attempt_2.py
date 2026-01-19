import pandas as pd

df = pd.read_csv('table.csv')
# Flatten the data to process all date entries
dates = []
for row in df.values:
    for date in row:
        dates.append(date)

# Count dates from 1990 or later
count = 0
for date in dates:
    year = int(date.split(', ')[1])
    if year >= 1990:
        count += 1

print(f"Final Answer: {count}")