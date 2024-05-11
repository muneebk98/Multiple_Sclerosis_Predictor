import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

def process_input(age, schooling, initial_symptom, oligoclonal_bands, llssep, ulssep,
                  vep, baep, periventricular_mri, cortical_mri, infratentorial_mri,
                  spinal_cord_mri, breastfeeding, varicella, mono_or_polysymptomatic, gender):
    # Create a data dictionary with input values
    input_dict = {
        'Age': float(age),
        'Schooling': float(schooling),
        'Initial_Symptom': float(initial_symptom),
        'Oligoclonal_Bands': float(oligoclonal_bands),
        'LLSSEP': float(llssep),
        'ULSSEP': float(ulssep),
        'VEP': float(vep),
        'BAEP': float(baep),
        'Periventricular_MRI': float(periventricular_mri),
        'Cortical_MRI': float(cortical_mri),
        'Infratentorial_MRI': float(infratentorial_mri),
        'Spinal_Cord_MRI': float(spinal_cord_mri),
        'Breastfeeding_2': 0 if breastfeeding == '1' else (1 if breastfeeding == '2' else 0),
        'Breastfeeding_3': 1 if breastfeeding == '3' else 0,
        'Varicella_2': 0 if varicella == '1' else (1 if varicella == '2' else 0),
        'Varicella_3': 1 if varicella == '3' else 0,
        'Mono_or_Polysymptomatic_2': 0 if mono_or_polysymptomatic == '1' else (1 if mono_or_polysymptomatic == '2' else 0),
        'Mono_or_Polysymptomatic_3': 1 if mono_or_polysymptomatic == '3' else 0,
        'Gender_2': 1 if gender == '2' else 0
    }
    return pd.DataFrame([input_dict])

def main():
    st.title("Multiple Sclerosis Diagnosis Tool")
    st.markdown("""
        <style>
        div.stButton > button:hover {
            background-color: deepskyblue !important;
            color: white;
        }
        </style>
        """, unsafe_allow_html=True)

    html_temp = """
    <div style="background-color:#025246; padding:10px">
    <h2 style="color:white;text-align:center;">Enter Your Symptoms</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    # Input fields with descriptions
    gender = st.selectbox("Gender (1 for Male, 2 for Female)", ['1', '2'])
    age = st.text_input("Age")
    schooling = st.text_input("Schooling (The total number of years the patient spent in educational institutions.)")
    breastfeeding = st.selectbox("Breastfeeding Status (1 for Yes, 2 for No, 3 for Unknown)", ['1', '2', '3'])
    varicella = st.selectbox("Varicella (Chickenpox) (1 for Positive, 2 for Negative, 3 for Unknown)", ['1', '2', '3'])
    initial_symptom = st.text_input("Initial Symptoms (Categorizes initial symptoms into types ranging from individual symptoms to combinations. Values range from 1 to 15)")
    mono_or_polysymptomatic = st.selectbox("Mono or Polysymptomatic Status (1 for Monosymptomatic, 2 for Polysymptomatic, 3 for Unknown)", ['1', '2', '3'])
    oligoclonal_bands = st.selectbox("Oligoclonal Bands (0 for Negative, 1 for Positive, 2 for Unknown)", ['0', '1', '2'])
    llssep = st.selectbox("LLSSEP (Lower Limb SSEP) (0 for Negative, 1 for Positive)", ['0', '1'])
    ulssep = st.selectbox("ULSSEP (Upper Limb SSEP) (0 for Negative, 1 for Positive)", ['0', '1'])
    vep = st.selectbox("VEP (Visual Evoked Potential) (0 for Negative, 1 for Positive)", ['0', '1'])
    baep = st.selectbox("BAEP (Brainstem Auditory Evoked Potential) (0 for Negative, 1 for Positive)", ['0', '1'])
    periventricular_mri = st.selectbox("Periventricular MRI (0 for Negative, 1 for Positive)", ['0', '1'])
    cortical_mri = st.selectbox("Cortical MRI (0 for Negative, 1 for Positive)", ['0', '1'])
    infratentorial_mri = st.selectbox("Infratentorial MRI (0 for Negative, 1 for Positive)", ['0', '1'])
    spinal_cord_mri = st.selectbox("Spinal Cord MRI (0 for Negative, 1 for Positive)", ['0', '1'])

    safe_html = """  
      <div style="background-color:#78A168;padding:10px">
       <h2 style="color:white;text-align:center;"> No MS  </h2>
       </div>
    """
    danger_html = """  
      <div style="background-color:#5C1515;padding:10px">
       <h2 style="color:white;text-align:center;"> MS </h2>
       </div>
    """

    if st.button("Diagnose"):
        # Assume process_input and model are defined as above and available here
        input_df = process_input(age, schooling, initial_symptom, oligoclonal_bands, llssep, ulssep,
                                 vep, baep, periventricular_mri, cortical_mri, infratentorial_mri,
                                 spinal_cord_mri, breastfeeding, varicella, mono_or_polysymptomatic, gender)
        output = model.predict(input_df)[0]

        if output == 1:
            st.markdown(danger_html, unsafe_allow_html=True)
        else:
            st.markdown(safe_html, unsafe_allow_html=True)

if __name__ == '__main__':
    main()