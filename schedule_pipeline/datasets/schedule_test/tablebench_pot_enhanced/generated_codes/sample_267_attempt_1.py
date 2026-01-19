import pandas as pd

df = pd.read_csv('table.csv')
# Filter for peaks in Morocco with elevation >= 3000 and col > 1500
filtered_peaks = df[(df['country'] == 'morocco') & 
                    (df['elevation (m)'].astype(int) >= 3000) & 
                    (df['col (m)'].astype(int) > 1500)]
# Calculate average prominence
avg_prominence = filtered_peaks['prominence (m)'].astype(int).mean()
print(f"Final Answer: {avg_prominence:.1f}")