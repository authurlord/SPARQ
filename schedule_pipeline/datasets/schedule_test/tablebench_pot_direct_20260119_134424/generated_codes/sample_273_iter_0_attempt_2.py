import pandas as pd

df = pd.read_csv('table.csv')
# Filter peaks in the Democratic Republic of the Congo with elevation >= 3000 meters
filtered_df = df[df['country'] == 'democratic republic of the congo']
filtered_df = filtered_df[filtered_df['elevation (m)'].astype(int) >= 3000]

# Calculate average prominence
average_prominence = filtered_df['prominence (m)'].astype(int).mean()

print(f"Final Answer: {average_prominence:.1f}")