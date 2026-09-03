import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Prediksi Risiko Stunting", page_icon="🧒", layout="centered")

# --- Load model & preprocessing artifacts (di-cache supaya tidak reload tiap interaksi) ---
@st.cache_resource
def load_artifacts():
    model = joblib.load('model_artifacts/stunting_model.pkl')
    numeric_imputer = joblib.load('model_artifacts/numeric_imputer.pkl')
    categorical_imputer = joblib.load('model_artifacts/categorical_imputer.pkl')
    encoder = joblib.load('model_artifacts/onehot_encoder.pkl')
    scaler = joblib.load('model_artifacts/scaler.pkl')
    metadata = joblib.load('model_artifacts/metadata.pkl')
    return model, numeric_imputer, categorical_imputer, encoder, scaler, metadata

model, numeric_imputer, categorical_imputer, encoder, scaler, metadata = load_artifacts()
numeric_features = metadata['numeric_features']
categorical_features = metadata['categorical_features']
best_threshold = metadata['best_threshold']

st.title("🧒 Prediksi Risiko Stunting pada Anak")
st.caption(f"Model: {metadata['model_name']} | Threshold keputusan: {best_threshold}")
st.write("Isi data anak di bawah ini untuk memprediksi risiko stunting berdasarkan indikator kesehatan dan sosial-ekonomi.")

with st.form("input_form"):
    st.subheader("Data Anak")
    col1, col2 = st.columns(2)
    with col1:
        age_months = st.number_input("Usia saat pengukuran (bulan)", 0, 60, 24)
        gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
        weight_kg = st.number_input("Berat badan saat ini (kg)", 1.0, 30.0, 11.0, step=0.1)
        height_cm = st.number_input("Tinggi/panjang badan saat ini (cm)", 30.0, 130.0, 80.0, step=0.1)
        head_circ = st.number_input("Lingkar kepala (cm)", 30.0, 55.0, 45.0, step=0.1)
        posture = st.selectbox("Posisi Pengukuran", ["Standing", "Lying"])
    with col2:
        birth_weight = st.number_input("Berat lahir (kg)", 0.5, 6.0, 3.0, step=0.1)
        birth_length = st.number_input("Panjang lahir (cm)", 30.0, 60.0, 49.0, step=0.1)
        birth_spacing = st.number_input("Jarak kelahiran dengan anak sebelumnya (bulan)", 0, 120, 24)
        exclusive_bf = st.selectbox("ASI Eksklusif (6 bulan pertama)", ["Yes", "No"])
        comp_feeding = st.selectbox("Kecukupan Protein Hewani (MPASI)", ["Adequate", "Inadequate"])
        immunization = st.selectbox("Status Imunisasi", ["Complete", "Incomplete"])
    infection_history = st.selectbox("Riwayat Infeksi", ["No_Record", "Yes", "No"])

    st.subheader("Data Ibu")
    col3, col4 = st.columns(2)
    with col3:
        mother_height = st.number_input("Tinggi badan ibu (cm)", 120.0, 180.0, 153.0, step=0.1)
        mother_preg_age = st.number_input("Usia ibu saat hamil (tahun)", 15, 50, 27)
        mother_anc = st.number_input("Jumlah kunjungan ANC", 0, 12, 4)
        mother_education = st.selectbox("Pendidikan Ibu", ["Primary_or_Lower", "Middle_School", "High_School", "Higher_Education"])
    with col4:
        mother_ced = st.selectbox("Status KEK Ibu", ["Yes", "No"])
        mother_iron = st.selectbox("Suplementasi Zat Besi", ["Yes", "No"])

    st.subheader("Lingkungan & Sosial-Ekonomi")
    col5, col6 = st.columns(2)
    with col5:
        clean_water = st.selectbox("Akses Air Bersih", ["Adequate", "Inadequate"])
        latrine = st.selectbox("Kepemilikan Jamban", ["With_Septic_Tank", "Without_Septic_Tank"])
        smoking = st.selectbox("Ada Anggota Keluarga Merokok", ["Yes", "No"])
    with col6:
        social_assist = st.selectbox("Penerima Bantuan Sosial", ["Yes", "No"])
        health_insurance = st.selectbox("Kepesertaan Asuransi Kesehatan", ["Yes", "No"])
        data_source = st.selectbox("Sumber Data", ["Posyandu", "Puskesmas", "Survey"])
        monitoring_type = st.selectbox("Jenis Pemantauan", ["Routine", "Intervention"])

    submitted = st.form_submit_button("Prediksi Risiko Stunting")

if submitted:
    raw_input = {
        'Age_at_Measurement_Months': age_months,
        'Gender': gender,
        'Weight_kg': weight_kg,
        'Height_or_Length_cm': height_cm,
        'Head_Circumference_cm': head_circ,
        'Measurement_Posture': posture,
        'Birth_Weight_kg': birth_weight,
        'Birth_Length_cm': birth_length,
        'Birth_Spacing_Months': birth_spacing,
        'Exclusive_Breastfeeding': exclusive_bf,
        'Complementary_Feeding_Animal_Protein': comp_feeding,
        'Immunization_Status': immunization,
        'Infection_History': infection_history,
        'Mother_Height_cm': mother_height,
        'Mother_Pregnancy_Age': mother_preg_age,
        'Mother_ANC_Visits': mother_anc,
        'Mother_Education': mother_education,
        'Mother_Chronic_Energy_Deficiency': mother_ced,
        'Mother_Iron_Supplementation': mother_iron,
        'Clean_Water_Access': clean_water,
        'Latrine_Ownership': latrine,
        'Family_Smoking_Habit': smoking,
        'Social_Assistance_Recipient': social_assist,
        'Health_Insurance_Status': health_insurance,
        'Data_Source': data_source,
        'Monitoring_Type': monitoring_type,
    }

    input_df = pd.DataFrame([raw_input])

    # Lengkapi kolom fitur yang belum terisi dari form (diisi NaN, akan ditangani imputer)
    for col in numeric_features + categorical_features:
        if col not in input_df.columns:
            input_df[col] = np.nan

    # --- Preprocessing (identik dengan pipeline saat training) ---
    input_num = numeric_imputer.transform(input_df[numeric_features])
    input_cat = categorical_imputer.transform(input_df[categorical_features])

    input_cat_encoded = encoder.transform(input_cat)
    input_num_scaled = scaler.transform(input_num)

    input_final = np.hstack([input_num_scaled, input_cat_encoded])

    # --- Prediksi ---
    proba = model.predict_proba(input_final)[0, 1]
    prediction = int(proba >= best_threshold)

    st.divider()
    st.subheader("Hasil Prediksi")
    st.metric("Probabilitas Risiko Stunting", f"{proba*100:.1f}%")

    if prediction == 1:
        st.error("⚠️ **Risiko Stunting Terdeteksi**")
        st.write("**Rekomendasi:** Segera lakukan verifikasi pengukuran ulang dan rujuk ke tenaga kesehatan / Puskesmas untuk pemeriksaan lebih lanjut.")
    else:
        st.success("✅ **Risiko Stunting Rendah**")
        st.write("**Rekomendasi:** Lanjutkan pemantauan rutin melalui Posyandu.")

    st.caption("Catatan: Hasil ini adalah prediksi model berbasis data, bukan diagnosis medis. Keputusan klinis tetap harus melalui tenaga kesehatan.")