import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import MinMaxScaler

current_dataframe = pd.read_csv("creditcard.csv")

fraud_cases_df = current_dataframe[current_dataframe['Class']==1]
normal_cases_df = current_dataframe[current_dataframe['Class']==0]

normal_cases_undersampled_df = normal_cases_df.sample(len(fraud_cases_df))

new_df = fraud_cases_df.append(normal_cases_undersampled_df)
correlations = new_df.corr()

#Correlation matrix
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,31,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(list(new_df.columns.values))
ax.set_yticklabels(list(new_df.columns.values))

#Weighing importance of features
array = new_df.values
X = array[:, 0:31]
Y = array[:, -1]
model = ExtraTreesClassifier()
model.fit(X, Y)
print("IMPORTANCE OF FEATURES:")
disp_df = pd.DataFrame({'Col': model.feature_importances_}, index=new_df.columns.values)
print(disp_df)

scaler = MinMaxScaler()
new_df = new_df[['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'Class']]
new_df[['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27']] = scaler.fit_transform(new_df[['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27']])

new_df.rename(columns={'Class': 'Fraud'}, inplace=True)
new_df['Normal'] = 1 - new_df['Fraud']

fraud_cases_df = new_df[new_df['Fraud']==1]
normal_cases_undersampled_df = new_df[new_df['Fraud']==0]

fraud_cases_train_df = fraud_cases_df.sample(frac=0.8)
normal_cases_undersampled_train_df = normal_cases_undersampled_df.sample(frac=0.8)

fraud_cases_train_df_indices = fraud_cases_train_df.index.tolist()
fraud_cases_test_df = fraud_cases_df[~fraud_cases_df.index.isin(fraud_cases_train_df_indices)]

normal_cases_undersampled_train_df_indices = normal_cases_undersampled_train_df.index.tolist()
normal_cases_undersampled_test_df = normal_cases_undersampled_df[~normal_cases_undersampled_df.index.isin(normal_cases_undersampled_train_df_indices)]

new_train_df = fraud_cases_train_df.append(normal_cases_undersampled_train_df)
new_test_df = fraud_cases_test_df.append(normal_cases_undersampled_test_df)

new_train_df.to_csv("creditcard_training_set.csv", index=False)
new_test_df.to_csv("creditcard_test_set.csv", index=False)

plt.show()