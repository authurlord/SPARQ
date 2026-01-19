import pandas as pd

df = pd.read_csv('table.csv')

# Define ionization energy columns
ie_columns = ['21st', '22nd', '23rd', '24th', '25th', '26th', '27th', '28th', '29th', '30th']

# Flag to track if we found the first IE > 700000
found = False
period = None

# Iterate over each row (element)
for _, row in df.iterrows():
    for col in ie_columns:
        value = row[col]
        if pd.notna(value) and value > 700000:
            period = col
            found = True
            break
    if found:
        break

print(f"Final Answer: {period}")