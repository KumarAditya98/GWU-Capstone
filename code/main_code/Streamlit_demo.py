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
            model = BlipForQuestionAnswering.from_pretrained("Model/blip-saved-model").to(device)
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

def main():
    st.balloons()
    title_text = 'Visual Question Answering (VQA) for Medical Scans'
    st.markdown(f"<h2 style='text-align: center;'><b>{title_text}</b></h2>", unsafe_allow_html=True)
    tab1, tab2 = st.tabs(['Test Data Demo','General Scan Demo'])
    with tab1:
        st.subheader(f"Benchmarking performance of {st.write('[CLEF](link to the dataset)')} dataset",)

    # Upload custom PyTorch model (.pt)
    selected_framework = st.radio("Select the Deep Learning framework:", ["PyTorch", "TensorFlow"])
    selected_model = st.radio("Indicate whether using custom model, pre-trained or pre-trained + custom head:", ["Custom", "Pre-trained","Pre-trained + Custom"])
    if selected_model in ["Pre-trained + Custom","Custom"]:
        model_file = st.file_uploader(
            f"Upload your {selected_framework} model (e.g., .pt for PyTorch, .h5 for TensorFlow)", type=["pt", "h5"],accept_multiple_files=False,
            help="""**Please include the import statement and if necessary to make sure the model exists on our server.
            Add custom heads to "Pre-trained + Custom" as shown below.
            Upload Format**:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '<your_library>']) 
            from tensorflow.keras.applications.xception import Xception
            model = Xception(weights='imagenet')
            (OR)
            model = models.resnet18(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, 10)""")
        if model_file and selected_framework == "PyTorch":
            if os.path.exists('model.pt'):
                os.remove('model.pt')
            with open(os.path.join(os.getcwd(), "model.pt"), "wb") as f:
                f.write(model_file.getbuffer())
        if model_file and selected_framework == "TensorFlow":
            if os.path.exists('model.h5'):
                os.remove('model.h5')
            with open(os.path.join(os.getcwd(), "model.h5"), "wb") as f:
                f.write(model_file.getbuffer())

    # Upload custom model architecture (.py)
    if selected_model == "Custom" and selected_framework == "PyTorch":
        model_architecture_code = st.text_area("Enter your custom model class if used PyTorch to create a custom model")
        st.code(model_architecture_code, language="python")
    elif selected_model == "Pre-trained + Custom" and selected_framework == "PyTorch":
        model_architecture_code = st.text_area("Instantiate pre-trained model with custom head. Note: write full library. TensorFlow as tf and torch as torch.")
        st.code(model_architecture_code, language="python")
    elif selected_model == "Pre-trained" and selected_framework in ['TensorFlow','PyTorch']:
        model_architecture_code = st.text_area("Instantiate pre-trained model with corresponding weights. Note: write full library. TensorFlow as tf and torch as torch.")
        st.code(model_architecture_code, language="python")
    else:
        model_architecture_code = True

    image_size = int(st.text_input("Enter the image size for your model (Note: For pre-trained models, it must match with image size that was used to train the model)", value="224"))
    Mean_list = (st.text_input("Enter your desired image normalization - Mean", value="0.5, 0.5, 0.5"))
    Std_list = (st.text_input("Enter your desired image normalization - Standard Deviation", value="0.5, 0.5, 0.5"))
    Mean_list = Mean_list.split(",")
    Std_list = Std_list.split(",")

    preprocess_fn_code = f"torchvision.transforms.Compose([torchvision.transforms.ToTensor(),\ntorchvision.transforms.Resize(({image_size}, {image_size})),\ntorchvision.transforms.Normalize(\nmean={[float(a) for a in Mean_list]},\nstd={[float(a) for a in Std_list]})])"
    st.text("Applied pre-processing")
    st.code(preprocess_fn_code, language="python")
    if model_architecture_code is not None and selected_model == "Custom" and selected_framework == 'PyTorch':
        clean_string = re.sub(r'#.*', '', model_architecture_code)
        clean_string = re.sub(r'(\'\'\'(.|\n)*?\'\'\'|"""(.|\n)*?""")', '', clean_string, flags=re.DOTALL)
        pattern = re.compile(r'class\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(.*\):',re.IGNORECASE)
        class_name = pattern.search(clean_string)
        if class_name:
            class_name = class_name.group(1)
    elif model_architecture_code is not None and selected_model in ["Pre-trained + Custom", 'Pre-trained']:
        model_name = st.text_input("The name of the model variable you've assigned (e.g model)", value='model')

    image_file = st.file_uploader("Upload the image you want to explain", type=["jpg", "jpeg", "png"])

    if model_architecture_code and image_file:
        if selected_framework == "PyTorch":
            if not isinstance(model_architecture_code, (bool)):
                exec(model_architecture_code, globals())
            # Load the PyTorch model
            if selected_model == "Pre-trained + Custom":
                # exec(model_architecture_code, globals())
                model = globals()[model_name]
                file = torch.load("model.pt", map_location=torch.device(device))
                model.load_state_dict(file)
            elif selected_model == "Custom":
                # exec(model_architecture_code, globals())
                model = globals()[model_name]
                file = torch.load("model.pt", map_location=torch.device(device))
                model.load_state_dict(file)
            else:
                model = globals()[model_name]
            model.eval()

        elif selected_framework == "TensorFlow":
            if not isinstance(model_architecture_code, (bool)):
                exec(model_architecture_code, globals())
            if selected_model == "Pre-trained":
                model = globals()[model_name]
            else:
                model = tf.keras.models.load_model("model.h5")

        else:
            st.error("Invalid framework selected.")

        # Load and display the image
        image = Image.open(image_file)
        image = image.resize((int(image_size), int(image_size)))
        st.image(image, caption="Uploaded Image", use_column_width=True)

        my_bool = True if selected_framework == "PyTorch" else False
        input_image = preprocess_image(image,preprocess_fn_code,image_size,PyTorch=my_bool,for_model = True)

        # Define a function for model prediction
        pred_orig = predict(input_image,selected_framework,model)
        st.write("Your Predicted Output from the model is as follows:", np.array(pred_orig))


        if st.button("Explain Model"):
            with st.spinner('Calculating LIME Analysis...'):
                if selected_framework == "PyTorch":
                    def batch_predict(images):
                        model.eval()
                        batch = torch.stack(tuple(preprocess_image(i,preprocess_fn_code,image_size=image_size,PyTorch = my_bool,for_model=False) for i in images), dim=0)
                        model.to(device)
                        batch = batch.to(device)
                        logits = model(batch)
                        probs = torch.nn.functional.softmax(logits, dim=1)
                        return probs.detach().cpu().numpy()
                    explainer = lime_image.LimeImageExplainer()
                    exp = explainer.explain_instance(np.array(image),
                                                 batch_predict,
                                                 top_labels=5,
                                                 hide_color=0,
                                                 num_samples=1000)
                else:
                    explainer = lime_image.LimeImageExplainer()
                    exp = explainer.explain_instance(np.array(input_image[0].astype('double')),
                                                     model.predict,
                                                     top_labels=5,
                                                     hide_color=0,
                                                     num_samples=1000)
                temp_1, mask_1 = exp.get_image_and_mask(exp.top_labels[0], positive_only=True,
                                                                    num_features=5, hide_rest=True)
                temp_2, mask_2 = exp.get_image_and_mask(exp.top_labels[0], positive_only=False,
                                                                    num_features=5, hide_rest=False)
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 15))
                ax1.imshow(mark_boundaries(temp_1, mask_1))
                ax2.imshow(mark_boundaries(temp_2, mask_2))
                ax1.axis('off')
                ax2.axis('off')
                plt.savefig('mask.png')
                Lime_img1 = Image.open('mask.png')
                st.image(Lime_img1)
                st.write("Image on the left denotes the super-pixels or region-of-interest based on LIME analysis. Classification is done due to the highlighted super-pixels. Image on the right imposes this region-of-interest on original image giving a more intuitive understanding.")
                dict_heatmap = dict(exp.local_exp[exp.top_labels[0]])
                heatmap = np.vectorize(dict_heatmap.get)(exp.segments)
                plt.imshow(heatmap, cmap='RdBu', vmin=-heatmap.max(), vmax=heatmap.max())
                plt.colorbar()
                plt.savefig('heatmap.png')
                Lime_img2 = Image.open('heatmap.png')
                st.image(Lime_img2)
                st.write("This section shows a heat-map that displays how important each super-pixel is to get some more granular explaianbility. The legend includes what color-coded regions of interest move the decision of the model. Blue indicates the regions that influences the decision of the model in the predicted class and red indicates the regions that influence the decision to other classes.")
