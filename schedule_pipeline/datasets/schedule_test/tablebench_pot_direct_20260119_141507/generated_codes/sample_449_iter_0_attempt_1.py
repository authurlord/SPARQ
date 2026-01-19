import pandas as pd

df = pd.read_csv('table.csv')
# Extract the 'length of term in days' column
term_days = df['length of term in days'].astype(float)

# Calculate mean and standard deviation
mean_term = term_days.mean()
std_term = term_days.std()

# Identify outliers using 2 standard deviations rule
threshold = 2 * std_term
anomalies = term_days[(term_days < (mean_term - threshold)) | (term_days > (mean_term + threshold))]

# Output the anomalies with their vice president names
anomalous_rows = df[(df['length of term in days'].astype(float) < (mean_term - threshold)) | 
                    (df['length of term in days'].astype(float) > (mean_term + threshold))]
print(f"Final Answer: {anomalous_rows[['vice president', 'length of term in days']].to_dict('records')}")