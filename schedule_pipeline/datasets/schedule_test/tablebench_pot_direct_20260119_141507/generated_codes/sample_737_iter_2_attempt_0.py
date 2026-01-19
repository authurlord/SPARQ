import pandas as pd

df = pd.read_csv('table.csv')
# Convert viewers (in millions) to numeric, handling any parsing issues
df['viewers (in millions)'] = pd.to_numeric(df['viewers (in millions)'], errors='coerce')

# Remove any NaN values (in case of parsing failure)
df = df.dropna(subset=['viewers (in millions)'])

# Extract viewership values as a list
viewership = df['viewers (in millions)'].tolist()

# Calculate the trend: check if it's increasing or decreasing
# We can use the last two differences to estimate the next value
if len(viewership) >= 2:
    # Compute differences between consecutive episodes
    diffs = [viewership[i+1] - viewership[i] for i in range(len(viewership)-1)]
    # If trend is increasing, average the differences and add to last value
    avg_diff = sum(diffs) / len(diffs) if diffs else 0
    next_viewership = viewership[-1] + avg_diff
    print(f"Final Answer: {next_viewership:.2f}")
else:
    print(f"Final Answer: {viewership[-1]:.2f}")