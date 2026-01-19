import pandas as pd

df = pd.read_csv('table.csv')

# Remove the 'TOTAL' row as it's a summary
df = df[df['Party'] != 'TOTAL']

# Initialize variables to track max increase and corresponding party
max_increase = float('-inf')
best_party = None

# Iterate over each party
for _, row in df.iterrows():
    first_duma = row['First Duma']
    fourth_duma = row['Fourth Duma']
    
    # Handle '–' or missing values
    if first_duma == '–' or fourth_duma == '–':
        continue
        
    # Convert to integers
    first_duma = int(first_duma.split()[0])
    fourth_duma = int(fourth_duma.split()[0])
    
    # Skip if first Duma is 0 (division by zero)
    if first_duma == 0:
        continue
        
    # Calculate percentage increase
    increase = ((fourth_duma - first_duma) / first_duma) * 100
    
    # Update max if current increase is higher
    if increase > max_increase:
        max_increase = increase
        best_party = row['Party']

print(f"Final Answer: {best_party}")