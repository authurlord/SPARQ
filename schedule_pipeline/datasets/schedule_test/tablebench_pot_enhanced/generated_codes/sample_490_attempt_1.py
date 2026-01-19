import pandas as pd

df = pd.read_csv('table.csv')

# Define the ionization energy columns
ie_columns = ['21st', '22nd', '23rd', '24th', '25th', '26th', '27th', '28th', '29th', '30th']

# Flag to track if we found the first IE > 700000
found = False

# Iterate over each column in order
for col in ie_columns:
    # Check all rows in this column
    for value in df[col]:
        if value is not None and value > 700000:
            print(f"Final Answer: {col}")
            found = True
            break
    if found:
        break