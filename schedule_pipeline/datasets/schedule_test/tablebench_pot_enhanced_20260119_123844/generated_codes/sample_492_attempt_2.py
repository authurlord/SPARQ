import pandas as pd

df = pd.read_csv('table.csv')

# Drop the header row since it's already in columns
df = df.iloc[1:]

# Convert relevant columns to numeric
miloš_columns = ['Miloš Zeman', 'Miloš Zeman_1']
df['Miloš Zeman_total'] = df[miloš_columns].apply(lambda x: x.astype(int).sum(), axis=1)

# Calculate total deputies and senators per round
round_data = df.groupby('Round').agg({
    'Miloš Zeman_total': 'sum'
}).reset_index()

# For each round, calculate total deputies and senators from all candidates
# We assume the total deputies and senators per round are the sum of all deputies and senators across all candidates
# But since only Miloš Zeman's data is available, we need to infer total from other candidates' data

# Reconstruct total deputies and senators per round by combining all candidate data
total_deputies = []
total_senators = []

for _, row in df.iterrows():
    round_name = row['Round']
    # Extract deputies and senators for all candidates
    deputies_sum = sum([
        row['Václav Klaus'], row['Václav Klaus_1'],
        row['Jaroslava Moserová'], row['Jaroslava Moserová_1'],
        row['Miloš Zeman'], row['Miloš Zeman_1']
    ])
    senators_sum = sum([
        row['Václav Klaus'], row['Václav Klaus_1'],
        row['Jaroslava Moserová'], row['Jaroslava Moserová_1'],
        row['Miloš Zeman'], row['Miloš Zeman_1']
    ])
    # Since the values are repeated per row, we take one row per round
    # So we use the first occurrence of each round
    if round_name not in [r for r in total_deputies]:
        total_deputies.append(deputies_sum)
        total_senators.append(senators_sum)

# Create a dictionary of total deputies and senators per round
round_totals = {}
for idx, round_name in enumerate(df['Round'].unique()):
    round_totals[round_name] = {
        'total_deputies': total_deputies[idx],
        'total_senators': total_senators[idx]
    }

# Now compute Miloš Zeman's share per round
majority_round = None
for _, row in df.iterrows():
    round_name = row['Round']
    miloš_deputies = row['Miloš Zeman'] + row['Miloš Zeman_1']
    total_deputies = round_totals[round_name]['total_deputies']
    total_senators = round_totals[round_name]['total_senators']
    
    # Total seats = deputies + senators
    total_seats = total_deputies + total_senators
    miloš_seats = miloš_deputies + (row['Miloš Zeman_1'] if row['Miloš Zeman_1'] != '-' else 0)
    
    # Only consider valid data
    if isinstance(miloš_seats, str) or isinstance(total_seats, str):
        continue
        
    if miloš_seats > 0.5 * total_seats:
        majority_round = round_name
        break

print(f"Final Answer: {majority_round}")