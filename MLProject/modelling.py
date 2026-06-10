import pandas as pd
import mlflow
import os
import shutil
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Set eksperimen agar rapi di log GitHub Actions
mlflow.set_experiment("Kriteria_3_CICD")

def train_model():
    # Path disesuaikan karena dijalankan di dalam folder MLProject
    data_path = "ispu_dki1_preprocessing.csv"
    print(f"Membaca dataset: {data_path}")
    df = pd.read_csv(data_path)

    # Memisahkan Fitur dan Target
    X = df.drop('categori', axis=1)
    y = df['categori']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Memastikan folder model_output bersih sebelum diisi model baru
    if os.path.exists("model_output"):
        shutil.rmtree("model_output")

    # PERBAIKAN: Gunakan autolog sesuai instruksi reviewer di Kriteria 2
    # Ini otomatis menggantikan mlflow.log_param dan mlflow.log_metric
    mlflow.autolog()

    # MLflow Tracking (Lokal di dalam runner GitHub)
    with mlflow.start_run(run_name="GitHub_Actions_Run"):
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Proses fit akan otomatis dicatat oleh autolog
        model.fit(X_train, y_train)

        # Prediksi hanya untuk ditampilkan (di-print) di log terminal GitHub Actions
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Akurasi Model: {acc:.4f}")

        # MENYIMPAN MODEL LOKAL UNTUK DOCKER (SYARAT ADVANCE)
        # Autolog sudah menyimpan di mlruns, tapi kita butuh copy-nya di 'model_output' untuk Dockerfile
        mlflow.sklearn.save_model(model, "model_output")
        
        print("Model berhasil dilatih dengan autolog dan disimpan ke folder 'model_output'!")

if __name__ == "__main__":
    train_model()
