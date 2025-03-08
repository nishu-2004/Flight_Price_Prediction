import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import os
import joblib

df = pd.read_csv('Clean_Dataset.csv')
df = df.drop('Unnamed: 0', axis=1)

df_copy = df.copy()
for col in df_copy.select_dtypes(include=['object']).columns:
    df_copy[col] = df_copy[col].astype('category').cat.codes

df_copy['price'] = np.log1p(df_copy['price'])

y = df_copy['price']
X = df_copy.drop('price', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

joblib.dump(scaler, 'scaler.save')

def build_model():
    model = Sequential([
        Dense(256, activation='relu', kernel_regularizer=l2(1e-6), input_shape=(X_train.shape[1],)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(128, activation='relu', kernel_regularizer=l2(1e-6)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu', kernel_regularizer=l2(1e-6)),
        BatchNormalization(),
        Dense(1)
    ])
    return model

model = build_model()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='mean_squared_error',
    metrics=['mean_absolute_error']
)

checkpoint_path = "checkpoint.weights.h5"
checkpoint = ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

if os.path.exists(checkpoint_path):
    try:
        model.load_weights(checkpoint_path)
    except:
        print("Failed to load checkpoint. Starting from scratch.")

history = model.fit(
    X_train, y_train, epochs=200, batch_size=32,
    validation_data=(X_test, y_test), verbose=1,
    callbacks=[checkpoint, early_stopping, reduce_lr]
)

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

test_loss, test_mae = model.evaluate(X_test, y_test)
print(f"Test MSE: {test_loss:.4f}")
print(f"Test MAE: {test_mae:.4f}")

y_pred = np.expm1(model.predict(X_test))

for i in range(10):
    print(f"Actual: {np.expm1(y_test.iloc[i]):.2f}, Predicted: {y_pred[i][0]:.2f}")

model.save('model.keras')