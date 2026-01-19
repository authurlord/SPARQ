import pandas as pd

df = pd.read_csv('table.csv')

# Remove the 'total' row (which is the last row) and filter parties with at least 2 in agricultural panel
filtered_df = df[df['agricultural panel'] >= 2].drop(index=df[df['party'] == 'total'].index)

# Convert the 'total' column to numeric (in case of any string issues)
filtered_df['total'] = pd.to_numeric(filtered_df['total'], errors='coerce')

# Sum the total seats for qualifying parties
total_seats_qualifying = filtered_df['total'].sum()

# Total seats across all parties (from the total row)
total_seats_all = df.loc[df['party'] == 'total', 'total'].values[0]

# Calculate percentage
percentage = (total_seats_qualifying / total_seats_all) * 100

print(f"Final Answer: {total_seats_qualifying}, {percentage:.1f}")