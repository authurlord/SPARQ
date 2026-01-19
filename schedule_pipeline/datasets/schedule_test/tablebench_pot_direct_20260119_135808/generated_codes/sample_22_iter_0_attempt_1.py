import pandas as pd

df = pd.read_csv('table.csv')
# Sum the 'number of people' column to get total deported people
total_people = df['number of people'].sum()
print(f"Final Answer: {total_people}")