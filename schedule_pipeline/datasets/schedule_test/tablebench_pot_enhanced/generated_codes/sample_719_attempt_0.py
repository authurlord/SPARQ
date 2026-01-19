import pandas as pd

df = pd.read_csv('table.csv')
# Sort by elevation (m) in descending order
df_sorted = df.sort_values(by='elevation (m)', ascending=False)
# Get the top 3 peaks
top_3_peaks = df_sorted['peak'].head(3).tolist()
print(f"Final Answer: {', '.join(top_3_peaks)}")