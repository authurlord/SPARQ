import pandas as pd

df = pd.read_csv('table.csv')
# Filter for New Zealand mountains
nz_mountains = df[df['country'] == 'new zealand']
# Sort by elevation in descending order
top_3 = nz_mountains.sort_values(by='elevation (m)', ascending=False).head(3)
# Extract the peak names
top_3_peaks = top_3['peak'].tolist()
print(f"Final Answer: {', '.join(top_3_peaks)}")