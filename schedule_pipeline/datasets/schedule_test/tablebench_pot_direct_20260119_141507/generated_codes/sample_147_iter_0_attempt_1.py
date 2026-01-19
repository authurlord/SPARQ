import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Extract relevant columns
examinees = df['number of examinees'].astype(int)
pass_percentage = df['pass percentage'].str.replace('%', '').astype(float) / 100  # Convert percentage to decimal

# Create scatter plot
plt.figure(figsize=(8, 5))
plt.scatter(examinees, pass_percentage, color='blue')
plt.title('Relationship between Number of Examinees and Pass Percentage')
plt.xlabel('Number of Examinees')
plt.ylabel('Pass Percentage (as decimal)')
plt.grid(True)
plt.show()

# Since the question is about the effect, we can also compute the correlation
correlation = examinees.corr(pass_percentage)
print(f"Final Answer: {correlation:.3f}")