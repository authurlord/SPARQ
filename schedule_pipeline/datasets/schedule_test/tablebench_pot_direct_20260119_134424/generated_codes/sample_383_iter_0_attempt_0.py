import pandas as pd

df = pd.read_csv('table.csv')
# Flatten the data to process all date entries
dates = []
for col in df.columns:
    dates.extend(df[col].tolist())

# Count dates from 1990 or later
count = 0
for date_str in dates:
    year = int(date_str.split(', ')[1])
    if year >= 1990:
        count += 1

print(f"Final Answer: {count}")