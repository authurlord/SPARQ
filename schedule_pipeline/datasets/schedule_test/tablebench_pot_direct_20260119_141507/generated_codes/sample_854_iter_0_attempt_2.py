import pandas as pd

df = pd.read_csv('table.csv')
# Extract enrollment values and location names
enrollment = df['enrollment'].astype(float)
locations = df['location']

# Remove duplicates by location (in case of duplicates in data)
unique_locations = df.drop_duplicates(subset=['location'])
enrollments = unique_locations['enrollment'].astype(float)

# Find the minimum difference between any two distinct enrollments
enrollments_sorted = sorted(enrollments)
min_diff = float('inf')
pair = None

for i in range(len(enrollments_sorted)):
    for j in range(i + 1, len(enrollments_sorted)):
        diff = abs(enrollments_sorted[i] - enrollments_sorted[j])
        if diff < min_diff:
            min_diff = diff
            idx1, idx2 = i, j
            loc1, loc2 = unique_locations.iloc[idx1]['location'], unique_locations.iloc[idx2]['location']

print(f"Final Answer: {loc1}, {loc2}, {min_diff:.0f}")