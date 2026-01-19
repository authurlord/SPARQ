import pandas as pd

df = pd.read_csv('table.csv')

# Define the ionization energy columns
ie_columns = ['21st', '22nd', '23rd', '24th', '25th', '26th', '27th', '28th', '29th', '30th']

# Flag to track if we've found the first occurrence
found = False

# Iterate over each row
for index, row in df.iterrows():
    for col in ie_columns:
        value = row[col]
        if pd.notna(value) and value > 700000:
            print(f"Final Answer: {col}")
            found = True
            break
    if found:
        break