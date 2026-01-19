import pandas as pd

df = pd.read_csv('table.csv')

# Filter for the two locomotives
empress = df[df['Name'] == 'Empress of Austria']
cyclops = df[df['Name'] == 'Cyclops']

# Extract introduced and withdrawn years
empress_intro = int(empress['Introduced'].values[0])
empress_withdrawn = int(empress['Withdrawn'].values[0])
cyclops_intro = int(cyclops['Introduced'].values[0])
cyclops_withdrawn = int(cyclops['Withdrawn'].values[0])

# Calculate service years
empress_service = empress_withdrawn - empress_intro
cyclops_service = cyclops_withdrawn - cyclops_intro

# Calculate difference
difference = empress_service - cyclops_service
print(f"Final Answer: {difference}")