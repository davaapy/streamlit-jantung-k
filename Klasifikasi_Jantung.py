import pickle
import streamlit as st

model = pickle.load(open('Klasifikasi_JJJ.sav', 'rb'))

st.title('Estimasi Klasifikasi Jantung')

age = st.number_input('Masukan umur')
sex = st.number_input('Masukan Gender,laki laki (Input1),perempuan (Input2)')
cp = st.number_input(
    'Masukan atypical angina: nyeri dada tidak berhubungan dengan jantung')
trestbps = st.number_input('Masukan chest pain type')
chol = st.number_input('Masukan tekanan darah')
fbs = st.number_input('Masukan serum cholestoral dalam mg/dl')
restecg = st.number_input('Masukan fasting blood sugar')
thalach = st.number_input('Masukan sinyal detak jantung yang tidak normal')
exang = st.number_input('Masukan denyut jantung maksimum tercapai')
oldpeak = st.number_input('Masukan diinduksi angina')
slope = st.number_input('Masukan Depresi ST')
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
