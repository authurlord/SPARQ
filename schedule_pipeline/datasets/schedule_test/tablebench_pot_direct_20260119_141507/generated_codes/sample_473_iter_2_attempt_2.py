import pandas as pd

df = pd.read_csv('table.csv')

# Check for negative values in "Main Worker"
main_worker_values = df[df['Particulars'] == 'Main Worker']['Total'].values[0]
if main_worker_values < 0:
    print("Anomaly detected: Negative value in 'Main Worker' column ('-10' and '-5').")
    print("Possible explanation: Data entry error or typo; number of workers cannot be negative.")
else:
    print("No anomaly found in 'Main Worker'.")

Final Answer: Negative value in Main Worker, data entry error