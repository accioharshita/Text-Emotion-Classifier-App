import streamlit as st

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import altair as alt

import joblib

pipe_lr= joblib.load(open('text_emotion_classifier_pipeline.pkl', 'rb'))

def predict_emotions(docx):
    results= pipe_lr.predict([docx])
    return results[0]

def get_prediction_proba(docx):
    results= pipe_lr.predict_proba([docx])
    return results



def main():
    st.title('Emotion Classifier Application')
    st.subheader('Emotion in Text')

    with st.form(key='emotion_clf_form'):
        raw_text= st.text_area('Type Here')
        submit_text= st.form_submit_button(label= 'Submit')

    if submit_text:
        col1, col2= st.columns(2)

        prediction= predict_emotions(raw_text)
        probability= get_prediction_proba(raw_text)

        with col1:
            st.success('Original Text')
            st.write(raw_text)

            st.success('Prediction')
            st.write(prediction)
            st.write('Confidence:{}'.format(np.max(probability)))

        with col2:
            st.success('Prediction Probability')
            st.write(probability)
            proba_df= pd.DataFrame(probability, columns= pipe_lr.classes_)
            st.write(proba_df.T)
            proba_df_clean= proba_df.T.reset_index()
            proba_df_clean.columns= ['emotions', 'probability']

            fig = alt.Chart(proba_df_clean).mark_bar().encode(x='emotions',y='probability',color='emotions')
            st.altair_chart(fig,use_container_width=True)


if __name__=='__main__':
    main()