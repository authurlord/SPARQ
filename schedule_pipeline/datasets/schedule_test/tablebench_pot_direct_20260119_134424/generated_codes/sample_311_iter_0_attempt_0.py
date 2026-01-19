import pandas as pd

df = pd.read_csv('table.csv')

# Initialize variables to store result
result_player = None
result_year = None

# Loop through each row
for index, row in df.iterrows():
    men_singles = row["men 's singles"]
    men_doubles = row["men 's doubles"]
    
    # Check if men's singles winner is in men's doubles (split by space and check)
    if men_singles in men_doubles:
        result_player = men_singles
        result_year = row['year']
        break

print(f"Final Answer: {result_player}, {result_year}")