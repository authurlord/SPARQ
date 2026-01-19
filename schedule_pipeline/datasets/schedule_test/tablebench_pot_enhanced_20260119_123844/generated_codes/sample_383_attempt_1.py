import pandas as pd

df = pd.read_csv('table.csv')
count = 0

# Iterate through each cell in the dataframe
for col in df.columns:
    for date_str in df[col]:
        # Extract year from the date string (e.g., 'march 28 , 1968' -> 1968)
        year = int(date_str.split(', ')[-1])
        if year >= 1990:
            count += 1

print(f"Final Answer: {count}")