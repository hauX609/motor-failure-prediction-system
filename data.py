import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import joblib
import os

base_dir = 'DataSet/CMAPSSData/'
training_files = ['train_FD001.txt', 'train_FD002.txt', 'train_FD003.txt', 'train_FD004.txt']
all_df_list = []
columns = ['engine_id', 'cycle', 'setting1', 'setting2', 'setting3'] + [f's{i}' for i in range(1, 22)]

for file_name in training_files:
    file_path = os.path.join(base_dir, file_name)
    df = pd.read_csv(file_path, sep='\s+', header=None, names=columns)
    all_df_list.append(df)

df_train = pd.concat(all_df_list, ignore_index=True)
print("\nAll data loaded. Shape:", df_train.shape)


# --- Preprocessing and Label Creation ---
constant_cols = [col for col in df_train.columns if df_train[col].nunique() == 1]
df_train.drop(columns=constant_cols, inplace=True)
print(f"Dropped constant columns: {constant_cols}")

max_cycles = df_train.groupby('engine_id')['cycle'].max().reset_index()
max_cycles.columns = ['engine_id', 'max_cycle']
df_train = pd.merge(df_train, max_cycles, on='engine_id', how='left')
df_train['RUL'] = df_train['max_cycle'] - df_train['cycle']

rul_threshold_critical = 15
rul_threshold_degrading = 50

def create_label(rul):
    if rul <= rul_threshold_critical: return 'Critical'
    elif rul <= rul_threshold_degrading: return 'Degrading'
    else: return 'Optimal'

df_train['status'] = df_train['RUL'].apply(create_label)

cols_to_exclude = ['engine_id', 'cycle', 'max_cycle', 'RUL', 'status']
cols_to_scale = [col for col in df_train.columns if col not in cols_to_exclude]
print(f"Columns to be scaled: {cols_to_scale}")

# Scale features
feature_scaler = MinMaxScaler()
df_train[cols_to_scale] = feature_scaler.fit_transform(df_train[cols_to_scale])

# Scale RUL labels
rul_scaler = MinMaxScaler()
df_train['RUL_scaled'] = rul_scaler.fit_transform(df_train[['RUL']])


# --- Create Sequences with TWO Labels ---
sequence_length = 50
sequences = []
labels_class = [] # For classification
labels_reg = []   # For regression

status_map = {'Optimal': 0, 'Degrading': 1, 'Critical': 2}
df_train['status_code'] = df_train['status'].map(status_map)

for engine_id in df_train['engine_id'].unique():
    engine_df = df_train[df_train['engine_id'] == engine_id]
    for i in range(len(engine_df) - sequence_length):
        sequences.append(engine_df[cols_to_scale].iloc[i:i + sequence_length].values)
        labels_class.append(engine_df['status_code'].iloc[i + sequence_length])
        labels_reg.append(engine_df['RUL_scaled'].iloc[i + sequence_length])

X = np.array(sequences)
y_class = to_categorical(np.array(labels_class), num_classes=3)
y_reg = np.array(labels_reg)

print(f"\nShape of X (sequences): {X.shape}")
print(f"Shape of y_class (status labels): {y_class.shape}")
print(f"Shape of y_reg (RUL labels): {y_reg.shape}")

# Input Layer
input_layer = Input(shape=(X.shape[1], X.shape[2]))

# Shared LSTM Layers
shared_lstm = LSTM(units=100, return_sequences=True)(input_layer)
shared_lstm = Dropout(0.2)(shared_lstm)
shared_lstm = LSTM(units=50)(shared_lstm)
shared_lstm = Dropout(0.2)(shared_lstm)

# Branch 1: Classification Head
class_head = Dense(10, activation='relu')(shared_lstm)
class_output = Dense(3, activation='softmax', name='class_output')(class_head)

# Branch 2: Regression Head
reg_head = Dense(10, activation='relu')(shared_lstm)
reg_output = Dense(1,activation='relu', name='reg_output')(reg_head)

# Define the model with one input and two outputs
model = Model(inputs=input_layer, outputs=[class_output, reg_output])

# Compile the model with two different loss functions
model.compile(
    optimizer='adam',
    loss={'class_output': 'categorical_crossentropy', 'reg_output': 'mean_squared_error'},
    metrics={'class_output': 'accuracy', 'reg_output': 'mae'}
)

model.summary()
# Train the Model
print("\nStarting multi-output model training...")
history = model.fit(
    X,
    {'class_output': y_class, 'reg_output': y_reg},
    epochs=10,
    batch_size=64, 
    validation_split=0.2
)
print("✅ Model training complete.")

shap_background_sample = X[:100]
joblib.dump(shap_background_sample, 'shap_background.pkl')

# Save all the necessary files
model.save('motor_model_multi.keras')
joblib.dump(feature_scaler, 'scaler.pkl')
joblib.dump(rul_scaler, 'rul_scaler.pkl')
joblib.dump(cols_to_scale, 'feature_columns.pkl')

print("\nSHAP background sample saved as shap_background.pkl")
print("\nAll assets saved successfully to the Colab environment.")
print("Triggering download for the saved files...")
