import pandas as pd

df = pd.read_csv('table.csv')

# Convert 'tackles' column to float (already numeric in string form)
df['tackles'] = pd.to_numeric(df['tackles'], errors='coerce')

# Extract the relevant columns
solo = df['solo'].astype(float)
assisted = df['assisted'].astype(float)
sack = df['sack'].astype(float)
yards = df['yards'].astype(float)
tds = df["td 's"].astype(float)

# Calculate average of each factor
avg_solo = solo.mean()
avg_assisted = assisted.mean()
avg_sack = sack.mean()
avg_yards = yards.mean()
avg_tds = tds.mean()

# Find the factor with the highest average contribution
max_factor = max(avg_solo, avg_assisted, avg_sack, avg_yards, avg_tds)
if max_factor == avg_solo:
    answer = 'solo'
elif max_factor == avg_assisted:
    answer = 'assisted'
elif max_factor == avg_sack:
    answer = 'sack'
elif max_factor == avg_yards:
    answer = 'yards'
else:
    answer = 'td \'s'

print(f"Final Answer: {answer}")