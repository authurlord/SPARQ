import pandas as pd

df = pd.read_csv('table.csv')

# Initialize variables to store result
winner = None
year = None

# Iterate through each row
for index, row in df.iterrows():
    men_singles = row["men 's singles"]
    men_doubles = row["men 's doubles"]
    
    # Check if men's singles winner is in men's doubles winners
    if men_singles in men_doubles:
        winner = men_singles
        year = row['year']
        break

print(f"Final Answer: {winner}, {year}")