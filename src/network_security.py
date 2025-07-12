# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import RocCurveDisplay
from mlxtend.plotting import plot_confusion_matrix
import joblib

# %%
# Load Data
train_data = pd.read_csv('KDDTrain+.txt', sep=',', header=None)
test_data = pd.read_csv('KDDTest+.txt', sep=',', header=None)

# %%
# Column Names
columns = (['duration','protocol_type','service','flag','src_bytes','dst_bytes','land','wrong_fragment','urgent','hot',
            'num_failed_logins','logged_in','num_compromised','root_shell','su_attempted','num_root','num_file_creations',
            'num_shells','num_access_files','num_outbound_cmds','is_host_login','is_guest_login','count','srv_count',
            'serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate','srv_diff_host_rate',
            'dst_host_count','dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate',
            'dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate',
            'dst_host_srv_rerror_rate','attack','level'])

train_data.columns = columns
test_data.columns = columns

# %%
# Create Binary Target
train_data['attack_state'] = train_data['attack'].apply(lambda x: 0 if x == 'normal' else 1)
test_data['attack_state'] = test_data['attack'].apply(lambda x: 0 if x == 'normal' else 1)

# %%
# One-Hot Encoding
train_data = pd.get_dummies(train_data, columns=['protocol_type','service','flag'], prefix="", prefix_sep="")
test_data = pd.get_dummies(test_data, columns=['protocol_type','service','flag'], prefix="", prefix_sep="")

# Align both datasets to same columns
train_data, test_data = train_data.align(test_data, join='outer', axis=1, fill_value=0)

# %%
# Encode attack labels with train only
attack_LE = LabelEncoder()
attack_LE.fit(train_data['attack'])
train_data['attack'] = attack_LE.transform(train_data['attack'])

# Handle unseen labels in test set (like 'saint')
def safe_transform(label):
    return attack_LE.transform([label])[0] if label in attack_LE.classes_ else -1

test_data['attack'] = test_data['attack'].apply(safe_transform)

# Drop rows with invalid (-1) attack labels
test_data = test_data[test_data['attack'] != -1]

# %%
# Split features and targets
X_train = train_data.drop(['attack', 'level', 'attack_state'], axis=1)
y_train = train_data['attack_state']
X_test = test_data.drop(['attack', 'level', 'attack_state'], axis=1)
y_test = test_data['attack_state']

# %%
# Scale
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %%
# Evaluate function
def Evaluate(Model_Name, Model_Abb, X_test, Y_test):
    Pred_Value = Model_Abb.predict(X_test)
    Accuracy = metrics.accuracy_score(Y_test, Pred_Value)
    Sensitivity = metrics.recall_score(Y_test, Pred_Value)
    Precision = metrics.precision_score(Y_test, Pred_Value)
    F1_score = metrics.f1_score(Y_test, Pred_Value)
    Recall = metrics.recall_score(Y_test, Pred_Value)
    
    print('--------------------------------------------------')
    print(f'{Model_Name} Accuracy   = {np.round(Accuracy, 3)}')
    print(f'{Model_Name} Sensitivity = {np.round(Sensitivity, 3)}')
    print(f'{Model_Name} Precision  = {np.round(Precision, 3)}')
    print(f'{Model_Name} F1 Score   = {np.round(F1_score, 3)}')
    print(f'{Model_Name} Recall     = {np.round(Recall, 3)}')
    print('--------------------------------------------------')

    Confusion_Matrix = metrics.confusion_matrix(Y_test, Pred_Value)
    plot_confusion_matrix(Confusion_Matrix, class_names=['Normal', 'Attack'], figsize=(5.5,5), colorbar="blue")
    RocCurveDisplay.from_estimator(Model_Abb, X_test, Y_test)

# %%
# Grid Search (optional tuning)
def GridSearch(Model_Abb, Parameters, X_train, Y_train):
    Grid = GridSearchCV(estimator=Model_Abb, param_grid=Parameters, cv=3, n_jobs=-1)
    Grid_Result = Grid.fit(X_train, Y_train)
    return Grid_Result.best_estimator_

# %%
# Random Forest Training
params = {'max_depth': [5, 10, 15]}
RF = RandomForestClassifier(random_state=42)
RF_best = GridSearch(RF, params, X_train_scaled, y_train)

# %%
RF_best.fit(X_train_scaled, y_train)

# %%
Evaluate('Random Forest Classifier', RF_best, X_test_scaled, y_test)

# save model
joblib.dump(RF_best, "rf_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(X_train.columns.to_frame().T, "rf_features.pkl")  # To align input columns
print("model has been saved")
