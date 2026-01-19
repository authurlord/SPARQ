import pandas as pd

df = pd.read_csv('table.csv')

# Convert 'timeslot rank' to numeric, replacing 'n / a' with NaN
df['timeslot rank'] = pd.to_numeric(df['timeslot rank'], errors='coerce')

# Filter episodes with viewers >= 10 and timeslot rank <= 3
filtered_df = df[(df['viewers (m)'] >= 10) & (df['timeslot rank'] <= 3)]

# Ensure 'rating' is numeric and compute average
if filtered_df.empty:
    average_rating = 0
else:
    average_rating = filtered_df['rating'].mean()

print(f"Final Answer: {average_rating:.2f}")