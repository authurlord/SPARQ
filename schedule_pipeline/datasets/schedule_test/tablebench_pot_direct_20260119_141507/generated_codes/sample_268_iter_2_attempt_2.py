import pandas as pd

df = pd.read_csv('table.csv')

# Convert 'rating' column to numeric, handling any parsing errors
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')

# Filter episodes with viewership >= 10 million and timeslot rank <= 3
filtered_df = df[
    (df['viewers (m)'] >= 10) & 
    (df['timeslot rank'].astype(str).str.isdigit() & (df['timeslot rank'] <= 3))
]

# Ensure we only keep valid numeric ratings and compute mean
if not filtered_df.empty:
    avg_rating = filtered_df['rating'].mean()
else:
    avg_rating = 0

print(f"Final Answer: {avg_rating:.2f}")