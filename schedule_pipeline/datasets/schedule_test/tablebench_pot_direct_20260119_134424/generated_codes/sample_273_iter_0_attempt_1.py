import pandas as pd

df = pd.read_csv('table.csv')
# Filter for peaks in the Democratic Republic of the Congo with elevation >= 3000 meters
dr_congo_peaks = df[df['country'].str.contains('democratic republic of the congo', case=False)]
high_elevation_peaks = dr_congo_peaks[dr_congo_peaks['elevation (m)'].astype(int) >= 3000]
average_prominence = high_elevation_peaks['prominence (m)'].astype(int).mean()
print(f"Final Answer: {average_prominence:.1f}")