import pandas as pd

df = pd.read_csv('table.csv')
# Filter peaks in Morocco with elevation >= 3000 and col > 1500
filtered_peaks = df[
    (df['country'] == 'morocco') &
    (df['elevation (m)'] >= 3000) &
    (df['col (m)'] > 1500)
]
# Calculate average prominence
average_prominence = filtered_peaks['prominence (m)'].mean()
print(f"Final Answer: {average_prominence:.1f}")