<H1>Multimodal Fusion: Advancing Medical Visual Question Answering</H1>

![Proposed Architecture](https://github.com/KumarAditya98/GWU-Capstone/blob/main/code/component/ProposedArchitecture.png)

## OVERVIEW
This project explores the application of Visual Question Answering (VQA) technology, which combines computer vision and natural language processing, in the medical domain, specifically for analyzing radiology scans. VQA can facilitate medical decision-making and improve patient outcomes by accurately interpreting medical imaging, which requires specialized expertise and time. The paper proposes developing an advanced VQA system for medical datasets using the BLIP architecture from Salesforce, leveraging deep learning and transfer learning techniques to handle the unique challenges of medical/radiology images. The paper discusses the underlying concepts, methodologies, and results of applying the BLIP architecture and fine-tuning approaches for Visual Question Answering in the medical domain, highlighting their effectiveness in addressing the complexities of VQA tasks for radiology scans. Inspired by the BLIP architecture from Salesforce, we propose a novel multimodal fusion approach for medical visual question answering and evaluating its promising potential.
## Folder Structure
```
.
├── code
│   ├── component
│   ├── configs ── medical_data_preprocess.yml
│   └── main_code
│               ├── CustomArchitecture 
│               │            ├── config.py
│               │            ├── custom_image_question_answer.py
│               │            ├── dataset.py
│               │            ├── predict_custom_transformer.py
│               │            └── train_file.py                                                                                                    
│               ├── Discarded                                        
│               ├── Convolution-patch-embedding-blip.py
│               ├── Convolution-patch-embedding-blip-finetuned.py
│               ├── Convolution-patch-embedding-blip-predict.py
│               ├── encoder_decoder_blip_vision.py
│               ├── encoder_decoder_blip_vision.py
│               ├── predict_medical_blip.py
│               ├── predict_medical_GIT.py
│               ├── predict_medical_vilt.py
│               ├── Streamlit_demo.py
│               ├── train_medical_blip.py
│               ├── train_medical_GIT.py
│               └── train_medical_vilt.py
├── demo
│   └── fig
├── full_report
│   ├── Latex_report
│   │   
│   ├── Markdown_Report
│   └── Word_Report
                  ├── Report.docx
                  └── Report.pdf
├── presentation
└── research_paper
    ├── Latex
    │   └── Fig
    └── Word
```

To run the code, clone the repository using the below command:
```commandline
git clone https://github.com/KumarAditya98/GWU-Capstone.git
```

## Dataset
We have combined two data sources for this project namely,
- [ImageCLEF 2019](https://zenodo.org/records/10499039)
- [VQA-RAD](https://osf.io/89kps/)

To download the dataset from these sources, run the following bash command on your linux environment:
```commandline
bash medical_data.sh
```
The command above creates a folder named 'dataset' which contains subfolders named 'train', 'validation', and 'test'.

### Configuring the Path to Medical Data

To configure the path to your medical data in the code, follow these steps:

1. Open the file `code/configs/medical_preprocess.yml`.
2. Locate the variable named `medical_data_root`.
3. It is currently set to `//home/ubuntu/VQA/dataset/medical_data/`.
4. Modify this path from `//home/ubuntu/VQA/` to the corresponding directory in your Linux environment.

For example, if your project is located in `/home/yourusername/projects/VQA/`, you would change the path to `//home/yourusername/projects/VQA/dataset/medical_data/`.

### Creating Excel Sheets for DataLoader and Augmenting Images

To create the Excel sheets for DataLoader and augment the images, follow these steps:

1. Navigate to the main code directory by running the following command in your terminal:
    ```bash
    cd code/main_code
    ```

2. Run the `medical_data_preprocessing.py` script using the following command:
    ```bash
    python3 medical_data_preprocessing.py -aug=True
    ```
    This command will initiate the augmentation process, which may take some time to complete.

    **Note:** If you wish to create the Excel sheets without augmenting the images, you can run the script without the augmentation flag:
    ```bash
    python3 medical_data_preprocessing.py
    ```

3. Once the script completes execution, the Excel sheets and augmented images (if augmentation was enabled) will be generated.

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

Here's an overview of the proposed architecture, along with the key files:

- **config.py**: This file sets up the configurations for the new architecture's training process.

- **custom_image_question_answer.py**: Defines the complete architecture of the new model, drawing inspiration from the BLIP VQA Architecture. This architecture is then utilized in `train_file.py` to define the model.

- **dataset.py**: Defines the dataloader for the new architecture, facilitating data handling during training.

- **inference_custom.py**: Responsible for checking the model's results on a single image, aiding in inference tasks.

- **predict_custom_transformer.py**: This script runs the trained model on the test set, providing insights into the model's performance on unseen data.

- **train_file.py**: Manages the complete training routine of the proposed architecture, orchestrating the training process.

Feel free to explore these files to gain a deeper understanding of the proposed architecture and its implementation.


