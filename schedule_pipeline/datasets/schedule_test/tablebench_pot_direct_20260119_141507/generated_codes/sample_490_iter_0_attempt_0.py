import pandas as pd

df = pd.read_csv('table.csv')

# Define the columns that represent energy levels (from 21st to 30th)
energy_columns = ['21st', '22nd', '23rd', '24th', '25th', '26th', '27th', '28th', '29th', '30th']

# Filter rows where any value in energy_columns exceeds 700000
exceeding_rows = []
for index, row in df.iterrows():
    for col in energy_columns:
        if pd.notna(row[col]) and float(row[col]) > 700000:
            exceeding_rows.append((row['number'], row['name'], col))
            break

# If there are exceeding rows, find the first one by atomic number
if exceeding_rows:
    # Sort by atomic number (number) to get the earliest period
    exceeding_rows.sort(key=lambda x: int(x[0]))
    first_exceeding = exceeding_rows[0]
    final_period = first_exceeding[0]
    print(f"Final Answer: {final_period}")
else:
    print("Final Answer: none")