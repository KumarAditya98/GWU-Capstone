import os
from ruamel.yaml import YAML
import pandas as pd

CUR_DIR = os.getcwd()
CODE_DIR = os.path.dirname(CUR_DIR)
PARENT_FOLDER = os.path.dirname(CODE_DIR)
EXCEL_FOLDER = PARENT_FOLDER + os.sep + 'Excel'
CONFIG_FOLDER = CODE_DIR + os.sep + 'configs'
os.makedirs(EXCEL_FOLDER, exist_ok=True)
combined_data_excel_file = EXCEL_FOLDER  + os.sep + "combined_data.xlsx"
test_data_excel_file = EXCEL_FOLDER  + os.sep + "test_data.xlsx"
def get_dataframe(cur_data_folder, file_type ='Train'):
    data_list = []
    qa_pairs_file =cur_data_folder+os.sep+ 'All_QA_Pairs_{}.txt'.format(file_type.lower())
    image_folder = cur_data_folder+os.sep+'{}_images'.format(file_type)
    with open(qa_pairs_file, 'r') as file:
        for line in file:
            all_info = line.strip().split('|')
            all_info[0] = image_folder + os.sep +all_info[0] +'.jpg'
            if os.path.exists(all_info[0]):
                #print("The image file exists.")
                pass
            else:
                print(f"The image file {all_info[0]} does not exist.")
                continue
            all_info.append(file_type.lower())
            data_list.append(all_info)
    df = pd.DataFrame(data_list, columns=['image_path', 'question', 'answer','split'])
    return df
def get_test_dataframe(cur_data_folder):
    data_list = []
    qa_pairs_file =cur_data_folder+os.sep+ 'VQAMed2019_Test_Questions_w_Ref_Answers.txt'
    image_folder = cur_data_folder + os.sep + 'VQAMed2019_Test_Images'
    with open(qa_pairs_file, 'r') as file:
        for line in file:
            all_info = line.strip().split('|')
            all_info[0] = image_folder + os.sep +all_info[0] +'.jpg'

            if os.path.exists(all_info[0]):
                #print("The image file exists.")
                pass
            else:
                print(f"The image file {all_info[0]} does not exist.")
                continue

            data_list.append(all_info)
    df = pd.DataFrame(data_list, columns=['image_path','category', 'question', 'answer'])

    return df

def create_excel(config):
    root_data = config['medical_data_root']
    train_data_folder = root_data + 'train'#config['train_data']
    val_data_folder = root_data +  'val'#config['val_data']
    test_data_folder = root_data + 'test'#config['test_data']

    train_data = get_dataframe(train_data_folder,file_type='Train')
    val_data = get_dataframe(val_data_folder,file_type='Val')

    combined_df = pd.concat([train_data, val_data], ignore_index=True)
    combined_df.to_excel(combined_data_excel_file, index=False)
    print(f"{'=' * 5} train val excel saved as combined_data.xlsx {'=' * 5}")

    test_df = get_test_dataframe(test_data_folder)
    test_df.to_excel(test_data_excel_file, index=False)
    print(f"{'=' * 5}  test excel saved as test_data.xlsx {'=' * 5}")

def main():
    """

    :rtype: object
    """
    yaml = YAML(typ='rt')
    config_file = os.path.join(CONFIG_FOLDER + os.sep + "medical_data_preprocess.yml" )

    with open(os.path.join(config_file), 'r') as file:
        config = yaml.load(file)
    print(f"{'='*5} Creating excel for VQA {'='*5}")
    create_excel(config)
    print(f"{'=' * 5} Creating excel for VQA success{'=' * 5}")



if __name__ == "__main__":
    main()