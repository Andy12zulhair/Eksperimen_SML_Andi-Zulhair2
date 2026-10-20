"""
Automation script for data preprocessing - Diabetes Risk Dataset
Author: Andi Zulhair
Kriteria 1 - Eksperimen terhadap Dataset Pelatihan
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def automate_preprocessing(raw_data_path, output_clean_path):
    """
    Otomatisasi pembersihan dan transformasi data dari dataset mentah ke dataset siap latih.
    """
    print(f"[INFO] Membaca raw dataset dari: {raw_data_path}")
    if not os.path.exists(raw_data_path):
        raise FileNotFoundError(f"File raw dataset tidak ditemukan di: {raw_data_path}")
        
    df = pd.read_csv(raw_data_path)
    print(f"[INFO] Raw dataset shape: {df.shape}")
    
    # 1. Handling Missing Values
    # Pada kolom alcohol_consumption, missing value (NaN) merepresentasikan 'None' (tidak mengonsumsi alkohol)
    df['alcohol_consumption'] = df['alcohol_consumption'].fillna('None')
    
    # Drop rows if any other unexpected null exists
    df = df.dropna().reset_index(drop=True)
    
    # 2. Encoding Categorical Features
    gender_map = {'Female': 0, 'Male': 1, 'Other': 2}
    df['gender'] = df['gender'].map(gender_map).fillna(1)
    
    cities = [
        'Mumbai', 'Delhi', 'Bangalore', 'Hyderabad', 'Ahmedabad', 'Chennai', 
        'Kolkata', 'Surat', 'Pune', 'Jaipur', 'Lucknow', 'Kanpur', 
        'Nagpur', 'Indore', 'Thane', 'Bhopal', 'Visakhapatnam', 'Patna'
    ]
    city_map = {city: i for i, city in enumerate(cities)}
    df['city'] = df['city'].map(city_map).fillna(0)
    
    act_map = {'Sedentary': 0, 'Moderate': 1, 'Active': 2}
    df['physical_activity_level'] = df['physical_activity_level'].map(act_map).fillna(1)
    
    diet_map = {'Balanced': 0, 'Vegan': 1, 'Vegetarian': 2, 'Non-Vegetarian': 3}
    df['diet_type'] = df['diet_type'].map(diet_map).fillna(0)
    
    smoke_map = {'Current': 0, 'Former': 1, 'Never': 2}
    df['smoking_status'] = df['smoking_status'].map(smoke_map).fillna(2)
    
    alc_map = {'None': 0, 'Moderate': 1, 'High': 2}
    df['alcohol_consumption'] = df['alcohol_consumption'].map(alc_map).fillna(0)
    
    inc_map = {'Low': 0, 'Medium': 1, 'High': 2}
    df['income_bracket'] = df['income_bracket'].map(inc_map).fillna(1)
    
    # 3. Standardization (Scaling) untuk semua fitur numerik/prediktor
    target_col = 'diabetes_risk'
    feature_cols = [c for c in df.columns if c != target_col]
    
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    # 4. Simpan output preprocessed
    os.makedirs(os.path.dirname(output_clean_path), exist_ok=True)
    df.to_csv(output_clean_path, index=False)
    print(f"[INFO] Preprocessing berhasil! Data disimpan ke: {output_clean_path}")
    print(f"[INFO] Processed dataset shape: {df.shape}")
    return df

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    raw_path = os.path.join(base_dir, '..', 'namadataset_raw', 'diabetes_risk_raw.csv')
    clean_path = os.path.join(base_dir, 'namadataset_preprocessing', 'diabetes_risk_clean.csv')
    
    automate_preprocessing(raw_path, clean_path)
