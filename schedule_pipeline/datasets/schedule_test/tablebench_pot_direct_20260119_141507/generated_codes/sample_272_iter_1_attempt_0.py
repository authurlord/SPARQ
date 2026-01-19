import pandas as pd

df = pd.read_csv('table.csv')

# Convert agricultural panel column to integers
df['agricultural panel'] = df['agricultural panel'].astype(int)

# Filter parties with at least 2 seats in agricultural panel
filtered_parties = df[df['agricultural panel'] >= 2]

# Total seats held by these parties
total_seats_filtered = filtered_parties['total'].sum()

# Total seats across all parties
total_seats_all = df.loc[df['party'] != 'total', 'total'].sum()

# Calculate percentage
percentage = (total_seats_filtered / total_seats_all) * 100

print(f"Final Answer: {total_seats_filtered}, {percentage:.1f}")