import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Filter rows for 'all around' event
filtered_df = df[df['event'] == 'all around']

# Further filter for specific competitions
selected_rows = filtered_df[(filtered_df['competition'] == 'olympic trials') | 
                            (filtered_df['competition'] == 'us championships')]

# Convert qualifying score to numeric, replacing 'n / a' with NaN
selected_rows['qualifying score'] = pd.to_numeric(selected_rows['qualifying score'], errors='coerce')

# Extract qualifying scores for both events
olympic_qual = selected_rows[selected_rows['competition'] == 'olympic trials']['qualifying score']
us_champs_qual = selected_rows[selected_rows['competition'] == 'us championships']['qualifying score']

# Calculate the change (difference) in qualifying scores
changes = us_champs_qual - olympic_qual

# Compute average change
average_change = changes.mean() if not changes.empty else 0

print(f"Final Answer: {average_change:.3f}")