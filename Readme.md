<H1>Multimodal Fusion: Advancing Medical Visual Question Answering</H1>
![Proposed Architecture](https://github.com/KumarAditya98/GWU-Capstone/blob/main/code/component/ProposedArchitecture.png)

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




