import pandas as pd

df = pd.read_csv('table.csv')
# Extract the 'length of term in days' column
term_days = df['length of term in days'].astype(float)

# Calculate mean and standard deviation for context
mean_term = term_days.mean()
std_term = term_days.std()

# Identify outliers using 3 standard deviations rule
outliers = term_days[(term_days < mean_term - 3 * std_term) | (term_days > mean_term + 3 * std_term)]

# Print the outlier values and their corresponding rows
print("Anomalies detected:")
for idx, row in df.iterrows():
    if row['length of term in days'] in outliers:
        print(f"Rank: {row['rank by time in office']}, VP: {row['vice president']}, Days: {row['length of term in days']}, Explanation: {row['explanation']}")

Final Answer: 10000