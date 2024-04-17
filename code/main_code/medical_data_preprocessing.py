import os
from ruamel.yaml import YAML
import pandas as pd
import numpy as np
import imgaug.augmenters as iaa
from PIL import Image, ImageEnhance, ImageFilter
import random
import argparse

CUR_DIR = os.getcwd()
CODE_DIR = os.path.dirname(CUR_DIR)
PARENT_FOLDER = os.path.dirname(CODE_DIR)
EXCEL_FOLDER = PARENT_FOLDER + os.sep + 'Excel'
CONFIG_FOLDER = CODE_DIR + os.sep + 'configs'
os.makedirs(EXCEL_FOLDER, exist_ok=True)
combined_data_excel_file = EXCEL_FOLDER  + os.sep + "combined_data.xlsx"
augmented_data_excel_file = EXCEL_FOLDER  + os.sep + "combined_aug_data.xlsx"
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

def vqa_rad_setup(df_train, df_test, curr_data_folder):
    """
    This function is to combine another data source into the existing ImageCLEF dataset. This new dataset has been taken from "https://osf.io/89kps/"
    :param df: dataframe to append files to
    :param curr_data_folder: data folder to access relevant images and files
    :return: combined df for test or train/val
    """
    file_name = "VQA_RAD Dataset Public.json"
    image_folder_path = curr_data_folder + os.sep + "VQA_RAD Image Folder"
    df = pd.read_json(os.path.join(curr_data_folder,file_name))
    df['image_path'] = image_folder_path + os.sep + df.image_name
    desired_columns = ['image_path', 'question', 'answer']
    df = df[desired_columns]
    #df['question'] = df['question'].str.lower()
    #df['answer'] = df['answer'].str.lower()
    df = df.sample(frac=1, random_state=42)
    train_size = int(0.8 * len(df))
    train_df = df.iloc[:train_size].copy()
    test_df = df.iloc[train_size:].copy()
    train_df['split'] = 'train'
    test_df['category'] = 'VQA-RAD'

    final_train_df = pd.concat([df_train, train_df], ignore_index=True)
    final_test_df = pd.concat([df_test, test_df], ignore_index=True)

    return final_train_df, final_test_df

def augment_images(train_df, output_folder):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    augmented_data = []

    for index, row in train_df.iterrows():
        image_path = row['image_path']
        try:
            image = Image.open(image_path).convert('RGB')  # Open image using PIL and convert to RGB
        except Exception as e:
            print(f"Error processing image at {image_path}: {e}")
            continue  # Skip if there's an error loading the image

        # Blur
        blurred_image = image.filter(ImageFilter.BLUR)
        blurred_image_path = os.path.join(output_folder, f"{os.path.basename(image_path)[:-4]}_blur.jpg")
        blurred_image.save(blurred_image_path)
        augmented_data.append({
            'image_path': blurred_image_path,
            'question': row['question'],
            'answer': row['answer'],
            'split': row['split']
        })

        # Brightness
        brightness_factor = random.uniform(0.5, 1.5)  # Random brightness factor
        brightness_enhancer = ImageEnhance.Brightness(image)
        brightened_image = brightness_enhancer.enhance(brightness_factor)
        brightened_image_path = os.path.join(output_folder, f"{os.path.basename(image_path)[:-4]}_bright.jpg")
        brightened_image.save(brightened_image_path)
        augmented_data.append({
            'image_path': brightened_image_path,
            'question': row['question'],
            'answer': row['answer'],
            'split': row['split']
        })

        # Noise
        noise_factor = random.uniform(0.0, 0.1)  # Random noise factor
        noisy_image = np.array(image)
        noise = np.random.normal(scale=noise_factor, size=noisy_image.shape).astype(np.uint8)
        noisy_image += noise
        noisy_image = np.clip(noisy_image, 0, 255)
        noisy_image = Image.fromarray(noisy_image)
        noisy_image_path = os.path.join(output_folder, f"{os.path.basename(image_path)[:-4]}_noise.jpg")
        noisy_image.save(noisy_image_path)
        augmented_data.append({
            'image_path': noisy_image_path,
            'question': row['question'],
            'answer': row['answer'],
            'split': row['split']
        })

    augmented_df = pd.DataFrame(augmented_data)
    return augmented_df


def create_excel(config):
    root_data = config['medical_data_root']
    train_data_folder = root_data + 'train'#config['train_data']
    val_data_folder = root_data +  'val'#config['val_data']
    test_data_folder = root_data + 'test'#config['test_data']
    aug_folder = train_data_folder + os.sep + 'aug'
    train_data = get_dataframe(train_data_folder,file_type='Train')
    val_data = get_dataframe(val_data_folder,file_type='Val')
    test_df = get_test_dataframe(test_data_folder)
    if args.aug:
        print('Augmenting the images!')
        aug_data = augment_images(train_data, aug_folder)
        combined_aug_df = pd.concat([train_data, val_data, aug_data], ignore_index=True)
        final_combined_df_aug, final_test_df = vqa_rad_setup(combined_aug_df, test_df, root_data)
        final_combined_df_aug.to_excel(augmented_data_excel_file, index=False)
        print(f"{'=' * 5} train with augmented data val excel saved as combined_aug_data.xlsx {'=' * 5}")
    else:
        print('Augmentation not selected')
        combined_df = pd.concat([train_data, val_data], ignore_index=True)
        final_combined_df, final_test_df = vqa_rad_setup(combined_df, test_df, root_data)
        final_combined_df.to_excel(combined_data_excel_file, index=False)
        print(f"{'=' * 5} train val excel saved as combined_data.xlsx {'=' * 5}")
    
    final_test_df.to_excel(test_data_excel_file, index=False)
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
    parser = argparse.ArgumentParser()
    parser.add_argument('-aug', type=bool, default=False, help='Specify if augmentation is required')
    args = parser.parse_args()
    main()