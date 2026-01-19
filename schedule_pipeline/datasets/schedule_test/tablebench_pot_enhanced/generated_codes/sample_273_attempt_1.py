import pandas as pd

df = pd.read_csv('table.csv')
# Filter for peaks in the Democratic Republic of the Congo with elevation >= 3000 meters
drc_peaks = df[df['country'].str.contains('democratic republic of the congo', case=False)]
high_elevation_peaks = drc_peaks[drc_peaks['elevation (m)'].astype(int) >= 3000]
# Calculate average prominence
average_prominence = high_elevation_peaks['prominence (m)'].astype(int).mean()
print(f"Final Answer: {average_prominence:.1f}")