import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'number of people' to integer and calculate total
total_people = df['number of people'].astype(int).sum()
print(f"Final Answer: {total_people}")