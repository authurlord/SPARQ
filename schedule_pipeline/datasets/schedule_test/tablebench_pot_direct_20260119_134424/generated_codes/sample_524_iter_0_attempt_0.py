import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'introduced' to integer for comparison; handle '-' as NaN
df['introduced'] = pd.to_numeric(df['introduced'], errors='coerce')

# Filter aircraft introduced by 2008 and not retired before 2008
# Aircraft active in 2004: only A319 (introduced 2004)
a319_capacity = df[df['aircraft'] == 'airbus a319 - 100']['seating'].iloc[0]

# Aircraft active in 2008: A319 (2004), A320 (2008), A321 (2008, retired later)
# Exclude those retired before 2008
active_2008 = df[(df['introduced'] <= 2008) & ((df['retired'] == '-') | (df['retired'] >= 2008))]
total_capacity_2008 = active_2008['seating'].astype(int).sum()

# Total capacity change
capacity_change = total_capacity_2008 - int(a319_capacity)

print(f"Final Answer: {capacity_change}")