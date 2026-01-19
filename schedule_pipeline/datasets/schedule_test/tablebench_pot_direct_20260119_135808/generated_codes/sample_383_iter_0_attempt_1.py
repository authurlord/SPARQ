import pandas as pd

df = pd.read_csv('table.csv')

# Flatten the data to get all date strings
all_dates = []
for row in df.values:
    all_dates.extend(row)

# Count dates from 1990 or later
count = 0
for date_str in all_dates:
    year = int(date_str.split(', ')[1])
    if year >= 1990:
        count += 1

print(f"Final Answer: {count}")