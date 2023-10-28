import pickle
import streamlit as st

model = pickle.load(open('Klasifikasi_JJJ.sav', 'rb'))

st.title('Estimasi Klasifikasi Jantung')

age = st.number_input('Masukan umur', step=1, max_value=100, min_value=1)
sex = st.number_input(
    'Masukan Gender,laki laki (Input1),perempuan (Input2)',  step=1, max_value=2, min_value=1)
cp = st.number_input(
    'Masukan atypical angina: nyeri dada tidak berhubungan dengan jantung')
trestbps = st.number_input('Masukan chest pain type')
chol = st.number_input('Masukan tekanan darah %',
                       step=1, max_value=250, min_value=1)
fbs = st.number_input('Masukan nomer serum cholestoral dalam mg/dl')
restecg = st.number_input('Masukan fasting gula darah',
                          step=1, max_value=250, min_value=1)
thalach = st.number_input(
    'Masukan sinyal detak jantung yang tidak normal', min_value=20.0, step=0.1)
exang = st.number_input(
    'Masukan denyut jantung maksimum tercapai', step=1, max_value=250, min_value=1)
oldpeak = st.number_input('Masukan diinduksi angina')
slope = st.number_input('Masukan Depresi ysng pernah anda capai')
ca = st.number_input('Masukan jumlah pembuluh darah utama')
thal = st.number_input('Masukan hasil stres thalium')


predict = ''

if st.button(' Estimasi Klasifikasi Jantung'):
    predict = model.predict(
        [[age, sex, cp, trestbps, chol, fbs, restecg,
            thalach, exang, oldpeak, slope, ca, thal]]
    )
    st.write('Estimasi Klasifikasi Jantung Dalam Ponds: ', predict)
    st.write('Estimasi Klasifikasi Jantung   Dalam TES: ', predict*2000)
