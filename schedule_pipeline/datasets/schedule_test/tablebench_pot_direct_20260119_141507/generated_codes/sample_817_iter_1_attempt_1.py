import pandas as pd

df = pd.read_csv('table.csv')
# Extract the 'tumbling' column and convert to numeric, handling any potential parsing issues
tumbling_scores = pd.to_numeric(df['tumbling'], errors='coerce')
# Calculate the mean, ignoring any invalid entries (though none expected here)
mean_tumbling = tumbling_scores.mean()
print(f"Final Answer: {mean_tumbling:.1f}")