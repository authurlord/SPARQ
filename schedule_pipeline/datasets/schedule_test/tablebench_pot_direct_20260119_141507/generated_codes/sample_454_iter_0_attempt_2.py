import pandas as pd

df = pd.read_csv('table.csv')

# Identify outliers in viewers
viewers = df['viewers'].astype(int)
normal_viewers = viewers[(viewers >= 300000) & (viewers <= 500000)]
outlier_high = viewers[viewers > 2000000]
outlier_low = viewers[viewers < 300000]

# Check for unusual timeslot
normal_timeslot = df[df['timeslot'].str.contains('8:30 pm - 9:30 pm', case=False)]
unusual_timeslot = df[~df['timeslot'].str.contains('8:30 pm - 9:30 pm', case=False)]

print(f"Outliers in viewers: {outlier_high.tolist()}")
print(f"Low viewers: {outlier_low.tolist()}")
print(f"Unusual timeslot: {unusual_timeslot['title'].tolist()}")

Final Answer: 2000000, the glamorous life