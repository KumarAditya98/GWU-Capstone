```
├── code
      ├── component
      ├── configs ── medical_data_preprocess.yml
      └── main_code
               ├── CustomArchitecture 
               │            ├── config.py
               │            ├── custom_image_question_answer.py
               │            ├── dataset.py
               │            ├── predict_custom_transformer.py
               │            └── train_file.py                                                                                                    
               ├── Discarded                                        
               ├── Convolution-patch-embedding-blip.py
               ├── Convolution-patch-embedding-blip-finetuned.py
               ├── Convolution-patch-embedding-blip-predict.py
               ├── encoder_decoder_blip_vision.py
               ├── encoder_decoder_blip_vision.py
               ├── predict_medical_blip.py
               ├── predict_medical_GIT.py
               ├── predict_medical_vilt.py
               ├── Streamlit_demo.py
               ├── train_medical_blip.py
               ├── train_medical_GIT.py
               └── train_medical_vilt.py
```
## Files Overview
Here's a brief overview of the key files in this repository:

- ### Finetuning Files

  - **Convolution-patch-embedding-blip.py**: Contains the autoencoder architecture of the vision model, specifically the convolution patch embedding layer.

  - **Convolution-patch-embedding-blip-finetuned.py**: This script trains the complete BLIP VQA model by incorporating the weights of the previously trained autoencoder into the patch embedding layer.

  - **encoder_decoder_blip_vision.py**: This file is used to fine-tune the entire vision transformer block in BLIP. It adds a decoder block to the vision encoder.

  - **medical_data_preprocessing.py**: After running the provided shell script, this script is responsible for creating data Excel files for the train, validation, and test sets. It combines the ImageCLEF and VQA-RAD datasets and also includes augmentation functionality.

  - **predict_medical_blip.py**: This script runs predictions on the test set using the BLIP model fine-tuned on the medical dataset.

  - **predict_medical_GIT.py**: Executes predictions on the test set using the GIT model fine-tuned on the medical dataset.

  - **predict_medical_vilt.py**: Executes predictions on the test set using the Vilt model fine-tuned on the medical dataset.

- ### Proposed Architecture Files

  - **config.py**: This file sets up the configurations for the new architecture's training process.

  - **custom_image_question_answer.py**: Defines the complete architecture of the new model, drawing inspiration from the BLIP VQA Architecture. This architecture is then utilized in `train_file.py` to define the model.

  - **dataset.py**: Defines the dataloader for the new architecture, facilitating data handling during training.

  - **inference_custom.py**: Responsible for checking the model's results on a single image, aiding in inference tasks.

  - **predict_custom_transformer.py**: This script runs the trained model on the test set, providing insights into the model's performance on unseen data.

  - **train_file.py**: Manages the complete training routine of the proposed architecture, orchestrating the training process.

Feel free to explore these files to gain a deeper understanding of the proposed architecture and its implementation.
