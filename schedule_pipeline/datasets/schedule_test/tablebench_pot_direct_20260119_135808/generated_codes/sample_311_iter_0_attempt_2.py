import pandas as pd

df = pd.read_csv('table.csv')

# Initialize variables to store result
result_year = None
result_player = None

# Iterate through each row
for index, row in df.iterrows():
    men_singles = row["men 's singles"]
    men_doubles = row["men 's doubles"]
    
    # Check if the men's singles winner is in the men's doubles winners
    if men_singles in men_doubles:
        result_year = row['year']
        result_player = men_singles
        break

print(f"Final Answer: {result_player}, {result_year}")