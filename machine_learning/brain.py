import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import pickle
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve



# Import the target variable
from target import y # Ensure this is properly aligned with df
from target import da

tickers_input = str(da[1][0])
df = pd.read_csv(f"/Users/simonebucciol/Desktop/project/csv_trading_signal/{tickers_input}_signal.csv")   # for the dataframe with the trading signals

# Drop non-numeric columns
if 'date' in df.columns:
    df.drop(columns=['date'], inplace=True)


# Ensure the index of y matches df before proceeding
y = y.replace([np.inf, -np.inf], np.nan).dropna()
y = y.astype(int)  # Ensure it's integer-based

# Ensure `X` and `y` have the same index
X = df.loc[y.index]  # Align X with y

# Replace infinities with NaN
X.replace([np.inf, -np.inf], np.nan, inplace=True)

# Handle missing values in X using mean imputation
imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(X), columns=df.columns, index=y.index)

# Verify final shapes before splitting
print(f"Final X shape: {X.shape}, Final y shape: {y.shape}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=1000, random_state=42)
rf_model.fit(X_train, y_train)

# Save the model
model_file_path = f'/Users/simonebucciol/Desktop/project/machine_learning/file_brain/{tickers_input}_rf_model.pkl'
with open(model_file_path, "wb") as file:
    pickle.dump(rf_model, file)

# Make predictions
y_pred = rf_model.predict(X_test)
print(f"The predictions for this ticker are: {y_pred}")

# Model Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.4f}%")

# Classification Report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# ROC Curve and AUC
fpr, tpr, _ = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)
print(f"ROC AUC: {roc_auc:.4f}")

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# plt.xlim([0.0, -1.0])
# plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_pred)
plt.figure()
plt.plot(recall, precision, color='blue', lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()

# Feature Importance
importances = rf_model.feature_importances_
feature_names = X.columns
feature_importances = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
print(feature_importances.sort_values(by='Importance', ascending=False))

# Visualize Feature Importance
plt.figure(figsize=(10, 6))
plt.barh(feature_importances['Feature'], feature_importances['Importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance')
plt.show()

# Error Analysis
errors = y_test != y_pred
error_analysis = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred, 'Error': errors})
print(error_analysis[error_analysis['Error'] == True])

# Creiamo DataFrame per allineare le date e visualizzare le posizioni
df_train = pd.DataFrame({'Position': y_train}, index=X_train.index)
df_test = pd.DataFrame({'Real Position': y_test, 'Predicted Position': y_pred}, index=X_test.index)

# Ordiniamo i dati per data
df_train = df_train.sort_index()
df_test = df_test.sort_index()

# Creazione della figura con due subplots
fig, ax1 = plt.subplots(figsize=(12, 7))

# --- Grafico del Prezzo di Chiusura ---
ax1.plot(df['close'], label="Close Price", color='blue')

# --- Segnali Reali ---
# Posizioni Long reali (se Position == 1)
ax1.plot(df_test.index[df_test['Real Position'] == 1], df['close'].loc[df_test.index[df_test['Real Position'] == 1]], 
         marker="o", linestyle="", color="green", label="Real Long (Buy)", markersize=8)
# Posizioni Short reali (se Position == 0)
ax1.plot(df_test.index[df_test['Real Position'] == 0], df['close'].loc[df_test.index[df_test['Real Position'] == 0]], 
         marker="o", linestyle="", color="red", label="Real Short (Sell)", markersize=8)

# --- Segnali Predetti ---
# Posizioni Long predetti (se Predicted Position == 1)
ax1.plot(df_test.index[df_test['Predicted Position'] == 1], df['close'].loc[df_test.index[df_test['Predicted Position'] == 1]], 
         marker="x", linestyle="", color="orange", label="Predicted Long (Buy)", markersize=8)
# Posizioni Short predetti (se Predicted Position == 0)
ax1.plot(df_test.index[df_test['Predicted Position'] == 0], df['close'].loc[df_test.index[df_test['Predicted Position'] == 0]], 
         marker="x", linestyle="", color="purple", label="Predicted Short (Sell)", markersize=8)

# --- Grafico delle Posizioni Real vs Predette ---
# Creiamo un altro grafico che mostra le posizioni Real e Predette
ax2 = ax1.twinx()  # Crea un secondo asse y condiviso
ax2.step(df_test.index, df_test['Real Position'], label="Real Position", color="green", linestyle='--', alpha=0.7, linewidth=2)
ax2.step(df_test.index, df_test['Predicted Position'], label="Predicted Position", color="orange", linestyle='--', alpha=0.7, linewidth=2)

# Aggiunta di etichette, legenda e titolo
ax1.set_title("Close Price with Real and Predicted Buy/Sell Signals")
ax1.set_ylabel("Price")
ax1.set_xlabel("Date")
ax1.legend(loc="upper left")

ax2.set_ylabel("Position (0 = Short, 1 = Long)")
ax2.legend(loc="upper right")

plt.xticks(rotation=45)  # Ruota le date per migliore leggibilità
plt.tight_layout()  # Ottimizza la disposizione
plt.show()
