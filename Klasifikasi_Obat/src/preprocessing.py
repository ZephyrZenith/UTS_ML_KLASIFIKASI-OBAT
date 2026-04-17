import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

def prepare_data(file_path):
    print("Membaca dataset...")
    df = pd.read_csv(file_path)
    
    le_dict = {}
    # Kolom teks yang perlu diubah menjadi angka
    categorical_cols = ['Gender', 'Blood_Pressure', 'Cholesterol', 'Blood_Sugar', 'Liver_Function', 'Kidney_Function', 'Drug_Class']
    
    print("Melakukan encoding pada data teks...")
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le
        
    # Memisahkan fitur (X) dan target (y)
    X = df.drop('Drug_Class', axis=1)
    y = df['Drug_Class']
    
    # Membagi data untuk training (80%) dan testing (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Melakukan standarisasi data...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Menyimpan alat pengubah data ini agar bisa dipakai di aplikasi antarmuka nanti
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(le_dict, 'label_encoders.pkl')
    
    print("Persiapan data selesai!")
    return X_train, X_test, y_train, y_test