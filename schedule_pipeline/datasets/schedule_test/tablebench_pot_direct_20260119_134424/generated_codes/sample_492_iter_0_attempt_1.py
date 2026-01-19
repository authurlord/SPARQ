import pandas as pd

df = pd.read_csv('table.csv')

# Define a function to check if a value is numeric or not
def safe_int(x):
    try:
        return int(x)
    except:
        return 0

# Iterate through rounds to find the first majority win for Miloš Zeman
for index, row in df.iterrows():
    round_name = row['Round']
    if round_name == 'Round':
        continue  # Skip header row

    # Extract deputies and senators for Miloš Zeman
    mz_deputies = safe_int(row['Miloš Zeman'])
    mz_senators = safe_int(row['Miloš Zeman_1'])

    # Total deputies and senators for Miloš Zeman
    mz_total = mz_deputies + mz_senators

    # Total possible deputies and senators in the round
    # Sum all deputies and senators from all candidates
    total_deputies = sum(safe_int(row[col]) for col in ['Václav Klaus', 'Václav Klaus_1', 'Jaroslava Moserová', 'Jaroslava Moserová_1', 'Miloš Zeman', 'Miloš Zeman_1'])
    total_senators = sum(safe_int(row[col]) for col in ['Václav Klaus', 'Václav Klaus_1', 'Jaroslava Moserová', 'Jaroslava Moserová_1', 'Miloš Zeman', 'Miloš Zeman_1'] if col.endswith('_1'))
    
    # Total possible votes
    total_possible = total_deputies + total_senators

    # Check if Miloš Zeman has a majority (>50%)
    if total_possible > 0 and mz_total > total_possible * 0.5:
        print(f"Final Answer: {round_name}")
        break
else:
    print("Final Answer: None")