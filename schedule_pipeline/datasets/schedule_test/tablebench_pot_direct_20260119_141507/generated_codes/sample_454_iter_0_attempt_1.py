import pandas as pd

df = pd.read_csv('table.csv')

# Identify unusual patterns
# Check for non-standard timeslot
non_standard_timeslot = df[df['timeslot'] != '8:30 pm - 9:30 pm']

# Check for outliers in viewers (using IQR method or direct inspection)
viewers = df['viewers'].astype(int)
normal_viewers = viewers[viewers < 2000000]
outlier_viewers = viewers[viewers >= 2000000]

print(f"Non-standard timeslot episodes:\n{non_standard_timeslot[['title', 'timeslot', 'viewers']]}")
print(f"Outliers in viewers (>= 2,000,000):\n{outlier_viewers}")

# Final Answer: non_standard_timeslot, high_viewer_outlier
Final Answer: non_standard_timeslot, high_viewer_outlier