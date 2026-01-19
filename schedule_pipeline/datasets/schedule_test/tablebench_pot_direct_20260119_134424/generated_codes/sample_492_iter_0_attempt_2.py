import pandas as pd

df = pd.read_csv('table.csv')

# Drop the header row
df = df.drop(df.index[0])

# Convert relevant columns to numeric
miloš_deputies = pd.to_numeric(df['Miloš Zeman'], errors='coerce')
miloš_senators = pd.to_numeric(df['Miloš Zeman_1'], errors='coerce')

# Calculate total deputies and senators for each round
total_deputies = pd.to_numeric(df['Václav Klaus'], errors='coerce') + \
                pd.to_numeric(df['Václav Klaus_1'], errors='coerce') + \
                pd.to_numeric(df['Jaroslava Moserová'], errors='coerce') + \
                pd.to_numeric(df['Jaroslava Moserová_1'], errors='coerce') + \
                miloš_deputies + \
                miloš_senators

total_senators = pd.to_numeric(df['Václav Klaus_1'], errors='coerce') + \
                 pd.to_numeric(df['Václav Klaus_1'], errors='coerce') + \
                 pd.to_numeric(df['Jaroslava Moserová_1'], errors='coerce') + \
                 pd.to_numeric(df['Jaroslava Moserová_1'], errors='coerce') + \
                 miloš_senators + \
                 miloš_senators

# Total combined deputies and senators
total_combined = total_deputies + total_senators

# Calculate Miloš Zeman's share
miloš_share = (miloš_deputies + miloš_senators) / total_combined

# Find the first round where Miloš Zeman has more than 50% of the total
for idx, row in df.iterrows():
    round_num = row['Round']
    if pd.notna(miloš_deputies[idx]) and pd.notna(miloš_senators[idx]):
        if miloš_share[idx] > 0.5:
            print(f"Final Answer: {round_num}")
            break
else:
    print("Final Answer: None")