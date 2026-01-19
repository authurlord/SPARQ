import pandas as pd

df = pd.read_csv('table.csv')

# Drop the header row
df = df[df['Round'] != 'Round']

# Convert Round to numeric for sorting
df['Round'] = df['Round'].astype(str)

# Group by Round and sum the values for Miloš Zeman's deputies and senators
miloš_zeman = df.groupby('Round')[['Miloš Zeman', 'Miloš Zeman_1']].apply(lambda x: x.sum(numeric_only=True))

# Calculate total possible deputies and senators per round
total_deputies = df.groupby('Round')['Václav Klaus'].apply(lambda x: x.sum(numeric_only=True) + 
                                                             df.loc[df['Round'] == x.name, 'Václav Klaus_1'].sum(numeric_only=True) +
                                                             df.loc[df['Round'] == x.name, 'Jaroslava Moserová'].sum(numeric_only=True) +
                                                             df.loc[df['Round'] == x.name, 'Jaroslava Moserová_1'].sum(numeric_only=True) +
                                                             df.loc[df['Round'] == x.name, 'Miloš Zeman'].sum(numeric_only=True) +
                                                             df.loc[df['Round'] == x.name, 'Miloš Zeman_1'].sum(numeric_only=True))

# Calculate total for Miloš Zeman
miloš_total = miloš_zeman.sum(axis=1)

# Calculate percentage
percentage = (miloš_total / total_deputies) * 100

# Find first round where percentage > 50
majority_round = percentage[percentage > 50].index[0] if not percentage[percentage > 50].empty else None

print(f"Final Answer: {majority_round}")