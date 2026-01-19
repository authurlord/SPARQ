import pandas as pd

df = pd.read_csv('table.csv')

# Remove the header row (first row is column names)
df = df.iloc[1:]

# Convert relevant columns to numeric
df['Miloš Zeman'] = pd.to_numeric(df['Miloš Zeman'], errors='coerce')
df['Miloš Zeman_1'] = pd.to_numeric(df['Miloš Zeman_1'], errors='coerce')

# Initialize variables
majority_round = None

# Iterate through each row
for index, row in df.iterrows():
    round_num = row['Round']
    mz_deputies = row['Miloš Zeman']
    mz_senators = row['Miloš Zeman_1']
    
    # Skip if data is missing
    if pd.isna(mz_deputies) or pd.isna(mz_senators):
        continue
    
    # Calculate total deputies and senators for this round
    total_deputies = row['Václav Klaus'] + row['Václav Klaus_1'] + row['Jaroslava Moserová'] + row['Jaroslava Moserová_1'] + mz_deputies
    total_senators = row['Václav Klaus_1'] + row['Jaroslava Moserová_1'] + mz_senators
    
    # Check if Miloš Zeman has more than 50% in both deputies and senators
    if (mz_deputies > 0.5 * total_deputies) and (mz_senators > 0.5 * total_senators):
        majority_round = round_num
        break

print(f"Final Answer: {majority_round}")