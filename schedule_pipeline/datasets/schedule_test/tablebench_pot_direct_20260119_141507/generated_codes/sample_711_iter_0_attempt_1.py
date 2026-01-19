import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Convert the numeric columns to float and handle missing values
df = df.apply(pd.to_numeric, errors='coerce')

# Extract the values for 2007 and 2011
values_2007 = df['2007'].dropna()
values_2011 = df['2011'].dropna()

# Only consider schools that have both 2007 and 2011 data
school_names = df['School']
increases = []

for idx in range(len(df)):
    school = school_names.iloc[idx]
    val_2007 = df.loc[idx, '2007']
    val_2011 = df.loc[idx, '2011']
    
    if pd.notna(val_2007) and pd.notna(val_2011):
        increase = val_2011 - val_2007
        increases.append((school, increase))

# Find the school with the maximum increase
if increases:
    max_increase_school = max(increases, key=lambda x: x[1])[0]
    print(f"Final Answer: {max_increase_school}")
else:
    print("Final Answer: None")