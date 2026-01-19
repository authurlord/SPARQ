import pandas as pd

df = pd.read_csv('table.csv')
# Filter peaks in the Democratic Republic of the Congo with elevation >= 3000 meters
drc_peaks = df[df['country'].str.contains('democratic republic of the congo', case=False) & (df['elevation (m)'].astype(int) >= 3000)]
# Calculate average prominence
avg_prominence = drc_peaks['prominence (m)'].astype(int).mean()
print(f"Final Answer: {avg_prominence:.1f}")