import pandas as pd

df = pd.read_csv('table.csv')

# Filter data for Miloš Zeman
miloš_data = df[['Round', 'Miloš Zeman', 'Miloš Zeman_1']]

# Initialize variables
majority_round = None

# Iterate through each row
for index, row in miloš_data.iterrows():
    round_name = row['Round']
    deputies = row['Miloš Zeman']
    senators = row['Miloš Zeman_1']
    
    # Skip if data is missing
    if deputies == '-' or senators == '-':
        continue
        
    # Convert to integers
    deputies = int(deputies)
    senators = int(senators)
    
    # Total deputies and senators
    total_deputies = 0
    total_senators = 0
    
    # Sum up total deputies and senators across all candidates
    # From the table, the total deputies and senators can be derived from other rows
    # But since it's not directly given, we can infer by looking at the maximum values in each category
    
    # Instead, let's calculate the total deputies and senators by summing up the max values in each category
    # We will use the first occurrence of valid data for total deputies and senators
    if total_deputies == 0:
        total_deputies = max(df['Václav Klaus'].astype(str).replace('-', '0').astype(int).max(),
                            df['Václav Klaus_1'].astype(str).replace('-', '0').astype(int).max(),
                            df['Jaroslava Moserová'].astype(str).replace('-', '0').astype(int).max(),
                            df['Jaroslava Moserová_1'].astype(str).replace('-', '0').astype(int).max(),
                            df['Miloš Zeman'].astype(str).replace('-', '0').astype(int).max(),
                            df['Miloš Zeman_1'].astype(str).replace('-', '0').astype(int).max())
    
    if total_senators == 0:
        total_senators = max(df['Václav Klaus'].astype(str).replace('-', '0').astype(int).max(),
                            df['Václav Klaus_1'].astype(str).replace('-', '0').astype(int).max(),
                            df['Jaroslava Moserová'].astype(str).replace('-', '0').astype(int).max(),
                            df['Jaroslava Moserová_1'].astype(str).replace('-', '0').astype(int).max(),
                            df['Miloš Zeman'].astype(str).replace('-', '0').astype(int).max(),
                            df['Miloš Zeman_1'].astype(str).replace('-', '0').astype(int).max())
    
    # Calculate percentage of deputies and senators
    pct_deputies = deputies / total_deputies
    pct_senators = senators / total_senators
    
    # Check if both are > 50%
    if pct_deputies > 0.5 and pct_senators > 0.5:
        majority_round = round_name
        break

print(f"Final Answer: {majority_round}")