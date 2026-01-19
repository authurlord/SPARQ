import pandas as pd

df = pd.read_csv('table.csv')
count = 0

# Iterate over each cell in the DataFrame to extract the year from the date strings
for col in df.columns:
    for date_str in df[col]:
        year = int(date_str.split(', ')[1])  # Extract year from "month day, year"
        if year >= 1990:
            count += 1

print(f"Final Answer: {count}")