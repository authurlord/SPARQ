import pandas as pd

df = pd.read_csv('table.csv')

# Clean column names
df.columns = ['Round', 'Vaclav Klaus', 'Vaclav Klaus_1', 'Jaroslava Moserová', 'Jaroslava Moserová_1', 'Miloš Zeman', 'Miloš Zeman_1']

# Filter out the header row
df = df[df['Round'] != 'Round']

# Convert numeric columns to numeric, handling '-' as NaN
numeric_columns = ['Vaclav Klaus', 'Vaclav Klaus_1', 'Jaroslava Moserová', 'Jaroslava Moserová_1', 'Miloš Zeman', 'Miloš Zeman_1']
df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

# Group by Round and calculate total deputies and senators for Miloš Zeman
miloš_data = df.groupby('Round')[['Miloš Zeman', 'Miloš Zeman_1']].sum()

# Calculate total seats (deputies + senators)
total_deputies = df['Miloš Zeman'].sum()
total_senators = df['Miloš Zeman_1'].sum()
total_seats = total_deputies + total_senators

# Calculate percentage of total seats for each round
miloš_data['Total'] = miloš_data.sum(axis=1)
miloš_data['Percentage'] = (miloš_data['Total'] / total_seats) * 100

# Find the first round where Miloš Zeman got over 50%
majority_round = miloš_data[miloš_data['Percentage'] > 50].index[0] if not miloš_data[miloš_data['Percentage'] > 50].empty else None

print(f"Final Answer: {majority_round}")