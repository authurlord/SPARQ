import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Convert 'viewers' to numeric, handling any parsing issues
df['viewers'] = pd.to_numeric(df['viewers'], errors='coerce')

# Drop any rows with invalid viewership (in case of missing values)
df = df.dropna(subset=['viewers'])

# Predict viewership as average of last 3 episodes (episodes 6, 7, 8)
last_three_viewers = df['viewers'].tail(3)
predicted_viewers = int(last_three_viewers.mean())

# Predict BBC Three weekly ranking as the most recent ranking (episode 8)
predicted_ranking = df['bbc three weekly ranking'].iloc[-1]

print(f"Final Answer: {predicted_viewers}, {predicted_ranking}")