# ============================================================
# FRAUDLYTICS - PASO 4: Modelo Multimodal de IA
# TensorFlow/Keras y PyTorch
# ============================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ============================================================
# CARGA Y PREPARACIÓN DE DATOS
# ============================================================

print("📦 Cargando datos con texto...")
df = pd.read_csv("data/datos_con_texto.csv")
print(f"✅ Datos cargados: {df.shape[0]} filas")

# Datos numéricos
features_numericas = [col for col in df.columns if col.startswith("V")]
features_lsa = [col for col in df.columns if col.startswith("lsa_")]

X_num = df[features_numericas].values
X_txt = df[features_lsa].values
y = df["Class"].values

# Escalado
scaler = StandardScaler()
X_num = scaler.fit_transform(X_num)

# Split
X_num_train, X_num_test, X_txt_train, X_txt_test, y_train, y_test = train_test_split(
    X_num, X_txt, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✅ Train: {X_num_train.shape[0]} | Test: {X_num_test.shape[0]}")

# ============================================================
# MODELO 1 - TENSORFLOW/KERAS: Functional API Multimodal
# ============================================================

print("\n🧠 Construyendo modelo con TensorFlow/Keras...")
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Rama 1: datos numéricos
input_num = Input(shape=(X_num_train.shape[1],), name="entrada_numerica")
x1 = Dense(64, activation="relu")(input_num)
x1 = Dropout(0.3)(x1)
x1 = Dense(32, activation="relu")(x1)

# Rama 2: datos de texto
input_txt = Input(shape=(X_txt_train.shape[1],), name="entrada_texto")
x2 = Dense(64, activation="relu")(input_txt)
x2 = Dropout(0.3)(x2)
x2 = Dense(32, activation="relu")(x2)

# Fusión de ramas
fusion = Concatenate()([x1, x2])
output = Dense(16, activation="relu")(fusion)
output = Dense(1, activation="sigmoid", name="salida")(output)

modelo_keras = Model(inputs=[input_num, input_txt], outputs=output)
modelo_keras.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

modelo_keras.summary()

print("\n⚡ Entrenando modelo Keras...")
early_stop = EarlyStopping(patience=3, restore_best_weights=True)
historia = modelo_keras.fit(
    [X_num_train, X_txt_train], y_train,
    validation_split=0.2,
    epochs=20,
    batch_size=256,
    callbacks=[early_stop],
    verbose=1
)

loss, acc = modelo_keras.evaluate([X_num_test, X_txt_test], y_test, verbose=0)
print(f"✅ Keras - Loss: {loss:.4f} | Accuracy: {acc:.4f}")

modelo_keras.save("data/modelo_keras.h5")
print("✅ Modelo Keras guardado")

# ============================================================
# MODELO 2 - PYTORCH: DataLoader y Backpropagation manual
# ============================================================

print("\n🔥 Construyendo modelo con PyTorch...")
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR

# Dataset personalizado
class FraudDataset(Dataset):
    def __init__(self, X_num, X_txt, y):
        self.X_num = torch.FloatTensor(X_num)
        self.X_txt = torch.FloatTensor(X_txt)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_num[idx], self.X_txt[idx], self.y[idx]

train_dataset = FraudDataset(X_num_train, X_txt_train, y_train)
test_dataset = FraudDataset(X_num_test, X_txt_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

# Modelo multimodal PyTorch
class FraudModel(nn.Module):
    def __init__(self, num_features, txt_features):
        super(FraudModel, self).__init__()
        self.rama_num = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.rama_txt = nn.Sequential(
            nn.Linear(txt_features, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.clasificador = nn.Sequential(
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x_num, x_txt):
        out_num = self.rama_num(x_num)
        out_txt = self.rama_txt(x_txt)
        fusion = torch.cat([out_num, out_txt], dim=1)
        return self.clasificador(fusion).squeeze()

modelo_pytorch = FraudModel(X_num_train.shape[1], X_txt_train.shape[1])
criterio = nn.BCELoss()
optimizador = AdamW(modelo_pytorch.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = StepLR(optimizador, step_size=5, gamma=0.5)

print("\n⚡ Entrenando modelo PyTorch...")
for epoch in range(10):
    modelo_pytorch.train()
    total_loss = 0
    for x_num, x_txt, targets in train_loader:
        optimizador.zero_grad()
        predicciones = modelo_pytorch(x_num, x_txt)
        loss = criterio(predicciones, targets)
        loss.backward()
        optimizador.step()
        total_loss += loss.item()
    scheduler.step()
    print(f"Epoch {epoch+1}/10 - Loss: {total_loss/len(train_loader):.4f}")

# Evaluación PyTorch
modelo_pytorch.eval()
correctos = 0
total = 0
with torch.no_grad():
    for x_num, x_txt, targets in test_loader:
        predicciones = modelo_pytorch(x_num, x_txt)
        predicciones = (predicciones > 0.5).float()
        correctos += (predicciones == targets).sum().item()
        total += targets.size(0)

print(f"✅ PyTorch - Accuracy: {correctos/total:.4f}")
torch.save(modelo_pytorch.state_dict(), "data/modelo_pytorch.pth")
print("✅ Modelo PyTorch guardado")

print("\n🎉 Paso 4 completado exitosamente!")