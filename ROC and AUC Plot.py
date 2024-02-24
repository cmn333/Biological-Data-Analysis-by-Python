from sklearn.metrics import roc_auc_score

# Calculate ROC AUC for each class
y_score = best_rf_model.predict_proba(x_test)
roc_auc_scores = []

for i in range(len(target_name)):
    y_true_binary = (y_test == i).astype(int)
    roc_auc = roc_auc_score(y_true_binary, y_score[:, i])
    roc_auc_scores.append(roc_auc)

# Print the result
for i in range(len(target_name)):
    print(f"ROC AUC for class {target_name[i]}: {roc_auc_scores[i]}")

# ROC curve generated 
fpr, tpr, threshold = roc_curve((y_test == 1).astype(int), y_score[:, 1])

fig, ax = plt.subplots(figsize=(8,8))
plt.title('Receiver Operating Characteristic')
ax.plot(fpr, tpr, color='orange', label=f'AUC = {roc_auc_scores[1]:.2f}')
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
