import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'rating' column to float
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
# Count episodes with rating >= 5.3
count_episodes = df[df['rating'] >= 5.3].shape[0]
print(f"Final Answer: {count_episodes}")