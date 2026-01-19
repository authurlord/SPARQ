import pandas as pd

df = pd.read_csv('table.csv')
count = 0

# Iterate through each column and extract dates
for col in df.columns:
    for date_str in df[col]:
        # Extract year from the date string (e.g., 'june 10 , 1964' -> 1964)
        year = int(date_str.split(',')[-1].strip())
        if year >= 1990:
            count += 1

print(f"Final Answer: {count}")