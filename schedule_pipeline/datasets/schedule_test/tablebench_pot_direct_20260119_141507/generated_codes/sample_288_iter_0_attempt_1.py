import pandas as pd

df = pd.read_csv('table.csv')

# Filter rows for the specified time periods
filtered_rows = df[(df['date of sterilization'] == 'jul 1940 - jun 1950') | (df['date of sterilization'] == 'jul 1950 - jun 1960')]

# Select the relevant columns (age 20-29 and age 30-39)
age_20_29 = filtered_rows['age 20 - 29'].astype(int)
age_30_39 = filtered_rows['age 30 - 39'].astype(int)

# Calculate total
total = age_20_29.sum() + age_30_39.sum()
print(f"Final Answer: {total}")