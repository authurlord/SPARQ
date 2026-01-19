import pandas as pd

df = pd.read_csv('table.csv')
# Extract the 'tumbling' column and convert to numeric, handling any potential parsing issues
tumbling_scores = pd.to_numeric(df['tumbling'], errors='coerce')
# Calculate the mean, ignoring any invalid entries (though none appear here)
average_tumbling = tumbling_scores.mean()
print(f"Final Answer: {average_tumbling:.1f}")