import pandas as pd

df = pd.read_csv('table.csv')
# Remove the 'TOTAL' row
df = df[df['Party'] != 'TOTAL']

# Initialize variables to track max increase and corresponding party
max_increase = float('-inf')
best_party = None

# Iterate through each party
for _, row in df.iterrows():
    first_duma = row['First Duma']
    fourth_duma = row['Fourth Duma']
    
    # Skip if either value is missing
    if first_duma == '–' or fourth_duma == '–':
        continue
    
    # Convert to integers
    first_duma = int(first_duma.split()[0])
    fourth_duma = int(fourth_duma.split()[0])
    
    # Calculate percentage increase
    if first_duma == 0:
        continue  # Avoid division by zero
    increase = ((fourth_duma - first_duma) / first_duma) * 100
    
    if increase > max_increase:
        max_increase = increase
        best_party = row['Party']

print(f"Final Answer: {best_party}")