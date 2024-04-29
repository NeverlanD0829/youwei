import xlrd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error

# Read the Excel file
workbook = xlrd.open_workbook('单种报告fps01.xls')

# Initialize lists to store all true values and predicted values
true_values_all = []
predicted_values_all = []

# Iterate through all worksheets
for sheet in workbook.sheets():
    # Read the second column (predicted values) and the third column (true values) of the worksheet
    predicted_values = sheet.col_values(1, start_rowx=1)
    true_values = sheet.col_values(2, start_rowx=1)

    # Convert the values to NumPy arrays
    predicted_values = np.array(predicted_values)
    true_values = np.array(true_values)

    # Append true values and predicted values to the overall lists
    true_values_all.extend(true_values)
    predicted_values_all.extend(predicted_values)

# Calculate the overall R2 score
overall_r2 = r2_score(true_values_all, predicted_values_all)

# Calculate the overall MAE
overall_mae = mean_absolute_error(true_values_all, predicted_values_all)

# Calculate the Mean Relative Error (MRE)
true_values_all = np.array(true_values_all)  # Convert to NumPy array
predicted_values_all = np.array(predicted_values_all)  # Convert to NumPy array
mre = np.mean(np.abs((true_values_all - predicted_values_all) / true_values_all))

# Create an R2 plot
plt.figure(figsize=(8, 6))
plt.scatter(true_values_all, predicted_values_all, alpha=0.5)
plt.title(f'R2 Plot (Overall R2 = {overall_r2:.4f}, Overall MAE = {overall_mae:.4f}, MRE = {mre:.4f})')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.grid(False)

# Show the plot
plt.show()

# Output the overall R2 score, MAE, and MRE
print(f'Overall R2: {overall_r2:.4f}')
print(f'Overall MAE: {overall_mae:.4f}')
print(f'MRE: {mre:.4f}')
