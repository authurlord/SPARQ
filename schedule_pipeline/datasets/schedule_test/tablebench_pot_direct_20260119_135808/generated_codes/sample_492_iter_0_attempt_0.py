import pandas as pd

df = pd.read_csv('table.csv')

# Convert data to numeric, handling '-' as NaN
df.iloc[1:, [1, 2, 3, 4, 5, 6]] = df.iloc[1:, [1, 2, 3, 4, 5, 6]].apply(pd.to_numeric, errors='coerce')

# Initialize variables to track rounds
majority_round = None

# Iterate over rounds
for idx, row in df.iterrows():
    if row['Round'] == 'Round':
        continue  # Skip header row

    round_name = row['Round']
    mz_deputies = row['Miloš Zeman']
    mz_senators = row['Miloš Zeman_1']

    # Skip if data is missing
    if pd.isna(mz_deputies) or pd.isna(mz_senators):
        continue

    # Calculate total deputies and senators in this round
    total_deputies = df.loc[df['Round'] == round_name, ['Václav Klaus', 'Václav Klaus_1', 'Jaroslava Moserová', 'Jaroslava Moserová_1', 'Miloš Zeman']].sum().sum()
    total_senators = df.loc[df['Round'] == round_name, ['Václav Klaus_1', 'Václav Klaus_1', 'Jaroslava Moserová_1', 'Jaroslava Moserová_1', 'Miloš Zeman_1']].sum().sum()

    # Avoid division by zero
    if total_deputies == 0 or total_senators == 0:
        continue

    # Calculate percentages
    pct_deputies = (mz_deputies / total_deputies) * 100
    pct_senators = (mz_senators / total_senators) * 100

    # Check for majority win (both > 50%)
    if pct_deputies > 50 and pct_senators > 50:
        majority_round = round_name
        break

print(f"Final Answer: {majority_round}")