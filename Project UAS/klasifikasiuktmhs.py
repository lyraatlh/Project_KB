# Lyra Attallah Aurellia_F55123014
# Klasifikasi kelayakan keringanan UKT

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Fungsi 1: Memuat dataset
def load_dataset(file_path):
    print("\n=== Tahap 1: Memuat Dataset ===")
    data = pd.read_csv(file_path)
    print("Dataset berhasil dimuat. Berikut lima baris pertama:")
    print(data.head())
    return data

data = load_dataset('Project UAS/data-klasifikasi-ukt-mahasiswa/klasifikasimhs.csv')

# Fungsi 2: Preprocessing data
def preprocess_data(data):
    print("\n=== Tahap 2: Preprocessing Data ===")
    encoder = LabelEncoder()
    data['Pekerjaan Orang Tua'] = encoder.fit_transform(data['Pekerjaan Orang Tua'])

    # Memisahkan fitur dan target
    X = data.drop(columns=['Kelayakan Keringanan UKT'])
    y = data['Kelayakan Keringanan UKT']
    print("Data preprocessing selesai. Fitur dan target dipisahkan.")
    return X, y

X, y = preprocess_data(data)

# Membagi dataset
print("\n=== Tahap 3: Membagi Dataset ===")
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
print(f"Data Training: {X_train.shape}, Data Testing: {X_test.shape}, Data Validasi: {X_val.shape}")

# Fungsi 3: Melatih model
def train_model(X_train, y_train):
    print("\n=== Tahap 4: Melatih Model ===")
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)
    print("Model berhasil dilatih.")
    return rf_model

rf_model = train_model(X_train, y_train)

# Fungsi 4: Evaluasi model
def evaluate_model(model, X, y, dataset_name):
    print(f"\n=== Evaluasi Model pada {dataset_name} ===")
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    print(f"Akurasi pada {dataset_name}: {accuracy:.2f}")
    print("Laporan Klasifikasi:")
    print(classification_report(y, y_pred))
    return y_pred

# Evaluasi pada data testing
y_pred_test = evaluate_model(rf_model, X_test, y_test, "Data Testing")

# Evaluasi pada data validasi
y_pred_val = evaluate_model(rf_model, X_val, y_val, "Data Validasi")

# Fungsi 5: Visualisasi hasil
def plot_confusion_matrix(y_true, y_pred, title):
    print(f"\n=== {title} ===")
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    plt.title(title)
    plt.show()

# Visualisasi Confusion Matrix
plot_confusion_matrix(y_test, y_pred_test, "Confusion Matrix - Data Testing")
plot_confusion_matrix(y_val, y_pred_val, "Confusion Matrix - Data Validasi")

# Fungsi 6: Visualisasi pentingnya fitur
def plot_feature_importance(model, feature_names):
    print("\n=== Plot Pentingnya Fitur ===")
    importances = model.feature_importances_
    plt.figure(figsize=(8, 5))
    plt.barh(feature_names, importances, align='center')
    plt.xlabel('Pentingnya Fitur')
    plt.title('Random Forest Feature Importances')
    plt.show()

plot_feature_importance(rf_model, X.columns)
