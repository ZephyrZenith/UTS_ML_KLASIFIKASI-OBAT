from preprocessing import prepare_data
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, f1_score
import joblib
import numpy as np  # <-- Library tambahan untuk memperbaiki error

def run_training():
    X_train, X_test, y_train, y_test = prepare_data('dataset/patient_clinical_profiles_real_drugs.csv')
    
    # PERBAIKAN: Konversi format label secara paksa menjadi bilangan bulat (integer)
    # Ini akan mencegah CatBoost error di Python 3.13
    y_train = np.array(y_train, dtype=np.int32)
    y_test = np.array(y_test, dtype=np.int32)
    
    print("Mulai melatih CatBoost...")
    cat_model = CatBoostClassifier(verbose=0)
    cat_model.fit(X_train, y_train)
    
    print("Mulai melatih XGBoost...")
    xgb_model = xgb.XGBClassifier(eval_metric='mlogloss')
    xgb_model.fit(X_train, y_train)
    
    print("Mulai melatih LightGBM...")
    lgb_model = lgb.LGBMClassifier()
    lgb_model.fit(X_train, y_train)
    
    print("\n=== HASIL AKURASI ===")
    for name, model in zip(['CatBoost', 'XGBoost', 'LightGBM'], [cat_model, xgb_model, lgb_model]):
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average='weighted')
        print(f"{name} - Accuracy: {acc:.4f}, F1-Score: {f1:.4f}")
    
    joblib.dump(cat_model, 'catboost_model.pkl')
    print("\nModel CatBoost berhasil disimpan!")

if __name__ == "__main__":
    run_training()