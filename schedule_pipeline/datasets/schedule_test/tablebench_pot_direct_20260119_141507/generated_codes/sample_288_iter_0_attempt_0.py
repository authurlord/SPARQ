import pandas as pd

df = pd.read_csv('table.csv')

# Filter rows for the specified time periods
filtered_rows = df[(df['date of sterilization'] == 'jul 1940 - jun 1950') | 
                   (df['date of sterilization'] == 'jul 1950 - jun 1960')]

# Extract the relevant columns and sum for age 20-29 and 30-39
total = (filtered_rows['age 20 - 29'].astype(int) + 
         filtered_rows['age 30 - 39'].astype(int)).sum()

print(f"Final Answer: {total}")