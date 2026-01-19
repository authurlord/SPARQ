import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Drop the first row (header row)
df = df.iloc[1:]

# Extract the relevant columns for Miloš Zeman
milo_zeman_deputies = df['Miloš Zeman']
milo_zeman_senators = df['Miloš Zeman_1']

# Total votes for Miloš Zeman per round
milo_zeman_total = milo_zeman_deputies + milo_zeman_senators

# Total votes for all candidates in each round
# Sum of all deputies and senators in each row
total_votes_per_round = (
    df['Václav Klaus'] + df['Václav Klaus_1'] +
    df['Jaroslava Moserová'] + df['Jaroslava Moserová_1'] +
    milo_zeman_deputies + milo_zeman_senators
)

# Check for majority win (>50%)
for idx, row in df.iterrows():
    total_milo = row['Miloš Zeman'] + row['Miloš Zeman_1']
    total_all = (
        row['Václav Klaus'] + row['Václav Klaus_1'] +
        row['Jaroslava Moserová'] + row['Jaroslava Moserová_1'] +
        row['Miloš Zeman'] + row['Miloš Zeman_1']
    )
    if total_milo > 0.5 * total_all:
        print(f"Final Answer: {row['Round']}")
        break