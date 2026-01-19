import pandas as pd

df = pd.read_csv('table.csv')

# Remove the 'TOTAL' row
df = df[df['Party'] != 'TOTAL']

# Function to convert string to int, handling '–' as 0
def safe_int(x):
    return 0 if x == '–' else int(x)

# Extract first and fourth duma seats
first_duma_seats = df['First Duma'].apply(safe_int)
fourth_duma_seats = df['Fourth Duma'].apply(safe_int)

# Calculate percentage increase
percentage_increase = ((fourth_duma_seats - first_duma_seats) / first_duma_seats) * 100

# Find the party with maximum percentage increase
max_increase_idx = percentage_increase.idxmax()
party_with_max_increase = df.loc[max_increase_idx, 'Party']

print(f"Final Answer: {party_with_max_increase}")