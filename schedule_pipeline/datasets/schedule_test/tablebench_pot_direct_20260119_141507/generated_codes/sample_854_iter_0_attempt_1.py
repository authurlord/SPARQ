import pandas as pd

df = pd.read_csv('table.csv')
# Convert enrollment to numeric (some might be strings, but here all seem numeric)
df['enrollment'] = pd.to_numeric(df['enrollment'], errors='coerce')

# Remove any NaN values due to parsing issues
df = df.dropna(subset=['enrollment'])

# Sort by enrollment to help identify close values
df_sorted = df.sort_values('enrollment')

# Compute pairwise differences between all unique schools
n = len(df_sorted)
min_diff = float('inf')
closest_pair = None

for i in range(n):
    for j in range(i + 1, n):
        diff = abs(df_sorted['enrollment'].iloc[i] - df_sorted['enrollment'].iloc[j])
        if diff < min_diff:
            min_diff = diff
            # Store the university names
            uni1 = df_sorted['location'].iloc[i]
            uni2 = df_sorted['location'].iloc[j]
            closest_pair = (uni1, uni2)

print(f"Final Answer: {closest_pair[0]}, {closest_pair[1]}, {min_diff}")