import pandas as pd

df = pd.read_csv('table.csv')
count = 0
# Iterate over each cell in the DataFrame
for col in df.columns:
    for date_str in df[col]:
        year = int(date_str.split(', ')[1])
        if year >= 1990:
            count += 1
print(f"Final Answer: {count}")