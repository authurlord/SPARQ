import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows for the specified time periods
filtered_df = df[df['date of sterilization'].isin(['jul 1940 - jun 1950', 'jul 1950 - jun 1960'])]
# Sum the values for age groups 20-29 and 30-39
total_sterilizations = filtered_df['age 20 - 29'].astype(int).sum() + filtered_df['age 30 - 39'].astype(int).sum()
print(f"Final Answer: {total_sterilizations}")