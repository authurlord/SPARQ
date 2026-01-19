import pandas as pd

df = pd.read_csv('table.csv')
# Extract the 'length of term in days' column
term_days = df['length of term in days'].astype(float)

# Calculate mean and standard deviation
mean_days = term_days.mean()
std_days = term_days.std()

# Detect outliers using 3 standard deviations rule
threshold = 3
outliers = term_days[(term_days < (mean_days - threshold * std_days)) | (term_days > (mean_days + threshold * std_days))]

# Print the anomalous values and their corresponding vice presidents
anomalies = []
for idx, row in df.iterrows():
    if row['length of term in days'] in outliers:
        anomalies.append((row['vice president'], row['length of term in days']))

print(f"Final Answer: {anomalies}")