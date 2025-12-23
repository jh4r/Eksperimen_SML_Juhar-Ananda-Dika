import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_data(path):
    """
    Fungsi untuk memuat data dari file CSV.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File {path} tidak ditemukan.")
    
    print(f"Memuat data dari {path}...")
    df = pd.read_csv(path)
    return df

def preprocess_data(df):
    """
    Fungsi utama untuk membersihkan dan memproses data.
    Langkah:
    1. Handling Missing Values & Tipe Data
    2. Hapus Duplikat dan ID
    3. Encoding (Label dan One-Hot)
    4. Scaling (StandardScaler)
    """
    print("Memulai preprocessing...")

    # Handling Missing Values dan Tipe Data
    # Ubah TotalCharges ke numerik, error jadi NaN, lalu isi 0
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0)

    # Hapus Duplikat dan ID
    df = df.drop_duplicates()
    
    if 'customerID' in df.columns:
        df = df.drop(['customerID'], axis=1)

    # Encoding
    # Label Encoding untuk Target 'Churn'
    if 'Churn' in df.columns:
        le = LabelEncoder()
        df['Churn'] = le.fit_transform(df['Churn'])
        print("Churn berhasil di-encode (Yes=1, No=0).")

    # One-Hot Encoding untuk fitur kategorikal
    # Mengubah True/False menjadi 1.0/0.0
    df_clean = pd.get_dummies(df)
    df_clean = df_clean.astype(float)

    # Scaling
    scaler = StandardScaler()
    numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    
    # Pastikan kolom numerik ada sebelum scaling
    if all(col in df_clean.columns for col in numerical_features):
        df_clean[numerical_features] = scaler.fit_transform(df_clean[numerical_features])
        print("Scaling fitur numerik selesai.")
    else:
        print("Warning: Salah satu kolom numerik tidak ditemukan, scaling dilewati.")

    return df_clean

def save_data(df, output_path):
    """
    Menyimpan data hasil olahan ke CSV baru.
    """
    df.to_csv(output_path, index=False)
    print(f"Data bersih berhasil disimpan ke: {output_path}")

if __name__ == "__main__":
    # Konfigurasi path 
    # Sesuaikan path ini dengan struktur foldermu
    # Asumsi: script ini ada di folder 'preprocessing', data ada di folder luar
    INPUT_PATH = '../Telco-Customer-Churn_raw.csv' 
    OUTPUT_PATH = 'telco_churn_clean.csv'

    try:
        # 1. Load
        df_raw = load_data(INPUT_PATH)
        
        # 2. Preprocess
        df_clean = preprocess_data(df_raw)
        
        # 3. Save
        save_data(df_clean, OUTPUT_PATH)
        
        print("\n=== Proses Otomasi Selesai ===")
        print(f"Ukuran Data Akhir: {df_clean.shape}")
        
    except Exception as e:
        print(f"\nTerjadi Error: {e}")