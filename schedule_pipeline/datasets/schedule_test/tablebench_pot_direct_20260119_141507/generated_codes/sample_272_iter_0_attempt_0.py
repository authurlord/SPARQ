import pandas as pd

df = pd.read_csv('table.csv')

# Filter parties with at least 2 seats in agricultural panel
filtered_parties = df[df['agricultural panel'] >= 2]

# Exclude the total row (last row)
filtered_parties = filtered_parties.dropna(subset=['party'])

# Sum total seats for these parties
total_seats = filtered_parties['total'].sum()

# Total seats across all parties (from the last row)
total_all_seats = df.loc[df['party'] == 'total', 'total'].values[0]

# Calculate percentage
percentage = (total_seats / total_all_seats) * 100

print(f"Final Answer: {total_seats}, {percentage:.1f}")