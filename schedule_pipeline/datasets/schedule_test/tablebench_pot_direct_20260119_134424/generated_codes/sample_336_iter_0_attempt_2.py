import pandas as pd

df = pd.read_csv('table.csv')

# Find the year where the same athlete won both championships
for index, row in df.iterrows():
    senior_pga_winner = row['senior pga championship']
    senior_players_winner = row['senior players championship']
    
    # Check if the winners are the same (ignoring the "(x/x)" part for comparison)
    if (senior_pga_winner.split(' ')[0] == senior_players_winner.split(' ')[0]):
        print(f"Final Answer: {row['year']}")
        break