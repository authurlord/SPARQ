import pandas as pd

df = pd.read_csv('table.csv')
# Filter for mountains in DRC with elevation >= 3000 meters
filtered_df = df[
    (df['country'].str.contains('democratic republic of the congo', case=False, na=False)) &
    (df['elevation (m)'].astype(int) >= 3000)
]
# Calculate average prominence
average_prominence = filtered_df['prominence (m)'].mean()
print(f"Final Answer: {average_prominence:.1f}")