import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Filter rows for 'all around' event and relevant competitions
filtered_df = df[(df['event'] == 'all around') & 
                 (df['competition'].isin(['olympic trials', 'us championships']))]

# Convert qualifying score to numeric, replacing 'n / a' with NaN
filtered_df['qualifying score'] = pd.to_numeric(filtered_df['qualifying score'], errors='coerce')

# Calculate the difference in qualifying scores
differences = filtered_df.loc[filtered_df['competition'] == 'us championships', 'qualifying score'] - \
              filtered_df.loc[filtered_df['competition'] == 'olympic trials', 'qualifying score']

# Average the differences
average_change = differences.mean()

print(f"Final Answer: {average_change:.3f}")