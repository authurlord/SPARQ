import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows for 'Empress of Austria' and 'Cyclops'
empress = df[df['Name'] == 'Empress of Austria']
cyclops = df[df['Name'] == 'Cyclops']

# Extract introduced and withdrawn years
empress_intro = int(empress['Introduced'].values[0])
empress_withdraw = int(empress['Withdrawn'].values[0])
cyclops_intro = int(cyclops['Introduced'].values[0])
cyclops_withdraw = int(cyclops['Withdrawn'].values[0])

# Calculate years in service
empress_years = empress_withdraw - empress_intro
cyclops_years = cyclops_withdraw - cyclops_intro

# Calculate difference
difference = empress_years - cyclops_years
print(f"Final Answer: {difference}")