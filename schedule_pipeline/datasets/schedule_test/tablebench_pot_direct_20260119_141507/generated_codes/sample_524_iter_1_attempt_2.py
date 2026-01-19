import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Convert 'introduced' and 'retired' to numeric, treating '-' as NaN
df['introduced'] = pd.to_numeric(df['introduced'], errors='coerce')
df['retired'] = pd.to_numeric(df['retired'], errors='coerce')

# Define function to check if aircraft is in service in a given year
def in_service(year):
    def is_active(row):
        introduced = row['introduced']
        retired = row['retired']
        # If retired is NaN (i.e., '-', meaning not retired), then still in service
        if pd.isna(retired):
            return introduced <= year
        else:
            return introduced <= year and retired > year
    return is_active

# Total seating in 2004
seating_2004 = df[df.apply(in_service(2004), axis=1)]['seating'].sum()

# Total seating in 2008
seating_2008 = df[df.apply(in_service(2008), axis=1)]['seating'].sum()

# Change in total seating capacity
change = seating_2008 - seating_2004

print(f"Final Answer: {change}")