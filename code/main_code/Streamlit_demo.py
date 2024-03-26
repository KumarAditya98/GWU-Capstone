import streamlit as st
import streamlit.components.v1 as components
import torch
import os
import pandas as pd
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering
import torch
import numpy as np


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
CUR_DIR = os.getcwd()
CODE_DIR = os.path.dirname(CUR_DIR)
PARENT_FOLDER = os.path.dirname(CODE_DIR)
EXCEL_FOLDER = PARENT_FOLDER + os.sep + 'Excel'
if not os.path.exists(EXCEL_FOLDER):
    raise FileNotFoundError(f"The folder {EXCEL_FOLDER} does not exist. Load data and run preprocessing first!! Exiting the program.")
combined_data_excel_file = EXCEL_FOLDER  + os.sep + "combined_data.xlsx"
test_data_excel_file = EXCEL_FOLDER + os.sep + "test_data.xlsx"
xdf_testdata = pd.read_excel(test_data_excel_file)
xdf_data = pd.read_excel(combined_data_excel_file)
xdf_fulldata = pd.concat([xdf_testdata,xdf_data])


st.balloons()
title_text = 'Visual Question Answering (VQA) for Medical Scans'
st.markdown(f"<h2 style='text-align: center;'><b>{title_text}</b></h2>", unsafe_allow_html=True)
tab1, tab2 = st.tabs(['Test Data Demo','General Scan Demo'])
with tab1:
    st.subheader(f"Benchmarking performance on CLEF dataset")
    selected_model = st.selectbox("Select the model to run inference", ('BLIP', 'BLIP-FineTuned', 'Vilt', 'GIT'),key = 1)
    if selected_model == 'BLIP' or selected_model == 'BLIP-FineTuned':
        if selected_model == 'BLIP':
            processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
            model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
        else:
            processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
            model = BlipForQuestionAnswering.from_pretrained("Model/blip-saved-model")
        st.write('Architecture and short description of model')
        st.image(os.path.join(CODE_DIR,'component/BLIP-general-Architecture.png'))
        st.caption('''Pre-training model architecture and objectives of BLIP (same parameters have the same color). Multimodal mixture of encoder-decoder, a unified vision-language model which can operate in one of the three functionalities: (1) Unimodal encoder is trained with an image-text contrastive (ITC) loss to align the vision and language representations. (2) Image-grounded text encoder uses additional cross-attention layers to model vision-language interactions, and is trained with a image-text matching (ITM) loss to distinguish between positive and negative image-text pairs. (3) Image-grounded text decoder replaces the bi-directional self-attention layers with causal self-attention layers, and shares the same cross-attention layers and feed forward networks as the encoder. The decoder is trained with a language modeling (LM) loss to generate captions given images.
        ''')
    st.write("Generate a random sample from the Test file for CLEF dataset")
    # Full data or only test data?? Test data has only one question per image and no repetitions.
    if 'sample_data' not in st.session_state:
        st.session_state.sample_data = xdf_fulldata.sample(n=1, replace=True)
    if st.button("Generate data",):
        st.session_state.sample_data = xdf_fulldata.sample(n=1, replace=True)
    st.session_state.image_path = str(st.session_state.sample_data.image_path.iloc[0])
    st.image(st.session_state.image_path)
    st.session_state.questions = xdf_fulldata[xdf_fulldata["image_path"] == st.session_state.image_path].question.values
    st.session_state.answers = xdf_fulldata[xdf_fulldata["image_path"] == st.session_state.image_path].answer.values
    st.write("Possible questions from the dataset are:")
    for i in range(len(st.session_state.questions)):
        st.write(f"{i+1}. {st.session_state.questions[i].capitalize()}")
    st.session_state.question = st.text_input("Write down your question here",placeholder=st.session_state.questions[0],value = st.session_state.questions[0]).lower()
    if st.button('Generate response'):
        st.session_state.image = Image.open(st.session_state.image_path).convert("RGB")
        st.session_state.encoding = processor(st.session_state.image, st.session_state.question, return_tensors="pt")
        st.session_state.out = model.generate(**st.session_state.encoding)
        st.session_state.generated_text = processor.decode(st.session_state.out[0], skip_special_tokens=True)
        st.write(f"The generated response is: {st.session_state.generated_text}")
        if st.session_state.question.lower() in st.session_state.questions:
            index = np.where(st.session_state.questions == st.session_state.question)[0][0]
            st.write(f"The ground truth response is: {st.session_state.answers[index]}")

with tab2:
    st.subheader(f"Evaluating performance on medical scan from internet")
    selected_model = st.selectbox("Select the model to run inference", ('BLIP', 'BLIP-FineTuned', 'Vilt', 'GIT'),key = 2)
    if selected_model == 'BLIP':
        processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
        model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
    elif selected_model == 'BLIP-FineTuned':
        processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
        model = BlipForQuestionAnswering.from_pretrained("Model/blip-saved-model").to(device)
    image_file_tab2 = st.file_uploader("Upload medical image file to ask questions", type=["jpg", "jpeg", "png"])
    if image_file_tab2:
        st.image(image_file_tab2)
        question_tab2 = st.text_input("Write down your question here")
        image_tab2 = Image.open(image_file_tab2).convert('RGB')
        encoding_tab2 = processor(image_tab2, question_tab2, return_tensors="pt")
        out_tab2 = model.generate(**encoding_tab2)
        generated_text_tab2 = processor.decode(out_tab2[0], skip_special_tokens=True)
        if st.button("Generate response",key=3):
            st.write(f"The generated response is: {generated_text_tab2.capitalize()}")
