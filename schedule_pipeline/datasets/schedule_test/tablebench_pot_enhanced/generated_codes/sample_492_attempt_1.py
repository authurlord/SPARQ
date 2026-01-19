import pandas as pd

df = pd.read_csv('table.csv')

# Convert the relevant columns to numeric, replacing '-' with 0
miloš_deputies = pd.to_numeric(df['Miloš Zeman'], errors='coerce').fillna(0)
miloš_senators = pd.to_numeric(df['Miloš Zeman_1'], errors='coerce').fillna(0)

# Calculate total deputies and senators for Miloš Zeman per row
miloš_total = miloš_deputies + miloš_senators

# Calculate total deputies and senators for all candidates per row
total_deputies = pd.to_numeric(df['Václav Klaus'], errors='coerce').fillna(0) + \
                 pd.to_numeric(df['Václav Klaus_1'], errors='coerce').fillna(0) + \
                 pd.to_numeric(df['Jaroslava Moserová'], errors='coerce').fillna(0) + \
                 pd.to_numeric(df['Jaroslava Moserová_1'], errors='coerce').fillna(0) + \
                 miloš_deputies + miloš_senators

total_senators = pd.to_numeric(df['Václav Klaus_1'], errors='coerce').fillna(0) + \
                 pd.to_numeric(df['Václav Klaus_1'], errors='coerce').fillna(0) + \
                 pd.to_numeric(df['Jaroslava Moserová_1'], errors='coerce').fillna(0) + \
                 pd.to_numeric(df['Jaroslava Moserová_1'], errors='coerce').fillna(0) + \
                 miloš_senators

total_combined = total_deputies + total_senators

# Calculate percentage of total for Miloš Zeman
percentage = miloš_total / total_combined * 100

# Find the first round where Miloš Zeman has >50%
rounds = df['Round']
for i, round_name in enumerate(rounds):
    if percentage[i] > 50:
        print(f"Final Answer: {round_name}")
        break
else:
    print("Final Answer: None")