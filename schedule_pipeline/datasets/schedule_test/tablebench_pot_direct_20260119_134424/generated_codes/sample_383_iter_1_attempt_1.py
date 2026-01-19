import pandas as pd

df = pd.read_csv('table.csv')
count = 0

# Iterate over each column and extract date strings from odd-numbered rows (1st, 3rd, 5th, etc.)
for col in df.columns:
    for i in range(0, len(df), 2):  # Rows 0, 2, 4 (0-indexed)
        date_str = df[col].iloc[i]
        year = int(date_str.split(',')[-1].strip())
        if year >= 1990:
            count += 1

print(f"Final Answer: {count}")