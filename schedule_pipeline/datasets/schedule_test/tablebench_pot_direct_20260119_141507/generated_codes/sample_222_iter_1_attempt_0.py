import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Drop the first row (header)
df = df.drop(df.index[0])

# Convert relevant columns to numeric
df = df.apply(pd.to_numeric, errors='coerce')

# Extract the school years (columns)
school_years = df.columns[1:]  # Exclude 'school year'

# Extract the values for Latvian and Russian
latvian_data = df.loc[1, school_years]  # Row for 'latvian'
russian_data = df.loc[2, school_years]  # Row for 'russian'
total_data = df.loc[4, school_years]   # Row for 'total'

# Calculate proportions
proportion_latvian = latvian_data / total_data
proportion_russian = russian_data / total_data

# Check if proportions are stable or changing
# We'll compute the differences in proportions over time
diff_latvian = proportion_latvian.diff()
diff_russian = proportion_russian.diff()

# Determine which one has a larger change (more significant influence)
# We'll look at the absolute change over time and find which one has more variation
change_latvian = diff_latvian.abs().sum()
change_russian = diff_russian.abs().sum()

# Final answer: which one is more significantly influenced?
if change_latvian > change_russian:
    final_answer = "Latvian"
else:
    final_answer = "Russian"

print(f"Final Answer: {final_answer}")