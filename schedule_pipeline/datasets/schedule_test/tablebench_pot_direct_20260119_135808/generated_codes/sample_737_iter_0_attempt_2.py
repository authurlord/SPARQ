import pandas as pd

df = pd.read_csv('table.csv')
# Convert viewers to float for numerical operations
df['viewers (in millions)'] = pd.to_numeric(df['viewers (in millions)'])

# Calculate the differences between consecutive episodes
viewers = df['viewers (in millions)'].values
differences = [viewers[i+1] - viewers[i] for i in range(len(viewers)-1)]

# Calculate average change
avg_change = sum(differences) / len(differences)

# Predict next viewership
next_viewership = viewers[-1] + avg_change

print(f"Final Answer: {next_viewership:.2f}")