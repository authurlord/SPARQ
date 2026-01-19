import pandas as pd

df = pd.read_csv('table.csv')

# Drop the 'TOTAL' row
df = df[df['Party'] != 'TOTAL']

# Initialize variables to track max increase and corresponding party
max_increase = float('-inf')
best_party = ''

# Iterate over each party
for _, row in df.iterrows():
    first_duma = row['First Duma']
    fourth_duma = row['Fourth Duma']
    
    # Convert strings to numbers, handling '-' as 0
    first_val = int(first_duma.split()[0]) if first_duma != '–' else 0
    fourth_val = int(fourth_duma.split()[0]) if fourth_duma != '–' else 0
    
    # Skip if first_duma is 0 to avoid division by zero
    if first_val == 0:
        continue
        
    increase = ((fourth_val - first_val) / first_val) * 100
    
    if increase > max_increase:
        max_increase = increase
        best_party = row['Party']

print(f"Final Answer: {best_party}")