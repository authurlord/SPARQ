import pandas as pd

df = pd.read_csv('table.csv')
# Filter peaks in DRC with elevation >= 3000 meters
drc_peaks = df[df['country'].str.contains('democratic republic of the congo', case=False)]
high_peaks = drc_peaks[drc_peaks['elevation (m)'].astype(int) >= 3000]
# Calculate average prominence
avg_prominence = high_peaks['prominence (m)'].astype(int).mean()
print(f"Final Answer: {avg_prominence:.1f}")