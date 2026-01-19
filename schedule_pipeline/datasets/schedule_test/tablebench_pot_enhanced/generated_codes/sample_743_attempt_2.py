import pandas as pd

df = pd.read_csv('table.csv')

# Extract the relevant columns
viewers = df['viewers (in millions)'].astype(float)
avg_18_49 = df['18 - 49 average'].astype(float)

# Calculate the average decrease per season
viewers_decrease = viewers.diff().mean()
avg_18_49_decrease = avg_18_49.diff().mean()

# Predict season 9 values
predicted_viewers = viewers.iloc[-1] + viewers_decrease
predicted_avg_18_49 = avg_18_49.iloc[-1] + avg_18_49_decrease

print(f"Final Answer: {predicted_viewers:.2f}, {predicted_avg_18_49:.2f}")