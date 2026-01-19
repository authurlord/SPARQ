import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Convert the data to a DataFrame properly
# The first row is header, so we skip it
data = df.values[1:]  # Skip the first row (header)
columns = df.columns.tolist()
df_clean = pd.DataFrame(data, columns=columns)

# Calculate percentage of students learning Russian
df_clean['% learning in russian'] = (df_clean['russian'] / df_clean['total']) * 100

# Now, we analyze the trend of % learning in latvian and % learning in russian against total students
# Since both are proportions, they should not change with total students
# We check if the percentages change over time

# Extract the percentage columns and total students
latvian_pct = df_clean['% learning in latvian'].values
russian_pct = df_clean['% learning in russian'].values
total_students = df_clean['total'].values

# Check if the percentages change significantly over time
# Since proportions are independent of total, we expect them to be stable
# But we can still compute the differences

# Compute the change in percentage over years
latvian_change = [latvian_pct[i] - latvian_pct[0] for i in range(1, len(latvian_pct))]
russian_change = [russian_pct[i] - russian_pct[0] for i in range(1, len(russian_pct))]

# Since the values are very close to stable, we can say neither is significantly influenced
# But if we must pick, we look at the magnitude of change
max_latvian_change = max(abs(c) for c in latvian_change)
max_russian_change = max(abs(c) for c in russian_change)

# If Russian change is larger, it's more influenced
if max_russian_change > max_latvian_change:
    final_answer = "russian"
else:
    final_answer = "latvian"

print(f"Final Answer: {final_answer}, latvian")