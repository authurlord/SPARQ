import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows for age groups between '10 - 14' and '22 - 35'
filtered_df = df[(df['Age Group'] == '10 - 14') | (df['Age Group'] == '15 - 17') | (df['Age Group'] == '18 - 21') | (df['Age Group'] == '22 - 35')]
# Sum the 'Both Gender' column
total_individuals = filtered_df['Both Gender'].sum()
print(f"Final Answer: {total_individuals}")