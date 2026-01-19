import pandas as pd

df = pd.read_csv('table.csv')

# Remove the header row
df = df[df['Round'] != 'Round']

# Convert 'Round' to categorical for proper sorting
df['Round'] = pd.Categorical(df['Round'], categories=['1st', '2nd', '3rd'], ordered=True)
df = df.sort_values('Round')

# Process each round
majority_round = None

for _, row in df.iterrows():
    round_name = row['Round']
    mz_deputies = int(row['Miloš Zeman']) if row['Miloš Zeman'] != '-' else 0
    mz_senators = int(row['Miloš Zeman_1']) if row['Miloš Zeman_1'] != '-' else 0
    mz_total = mz_deputies + mz_senators

    # Get total deputies and senators across all candidates in this round
    total_deputies = int(row['Václav Klaus']) + int(row['Václav Klaus_1']) + \
                     int(row['Jaroslava Moserová']) + int(row['Jaroslava Moserová_1']) + \
                     mz_deputies + mz_senators
    total_senators = int(row['Václav Klaus_1']) + int(row['Václav Klaus_1']) + \
                     int(row['Jaroslava Moserová_1']) + int(row['Jaroslava Moserová_1']) + \
                     mz_senators + mz_senators

    total_combined = total_deputies + total_senators

    # Check if Miloš Zeman has more than 50% of total
    if mz_total > 0.5 * total_combined:
        majority_round = round_name
        break

print(f"Final Answer: {majority_round}")