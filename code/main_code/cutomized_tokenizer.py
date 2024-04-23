from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import os
import pandas as pd
CUR_DIR = os.getcwd()
CODE_DIR = os.path.dirname(CUR_DIR)
PARENT_FOLDER = os.path.dirname(CODE_DIR)
EXCEL_FOLDER = PARENT_FOLDER + os.sep + 'Excel'
CONFIG_FOLDER = CODE_DIR + os.sep + 'configs'
TOKENIZER_File = CONFIG_FOLDER + os.sep + 'qa_tokenizer.json'
if not os.path.exists(EXCEL_FOLDER):
    raise FileNotFoundError(f"The folder {EXCEL_FOLDER} does not exist. Load data and run preprocessing first!! Exiting the program.")
combined_data_excel_file = EXCEL_FOLDER  + os.sep + "combined_aug_data.xlsx"
xdf_data = pd.read_excel(combined_data_excel_file)
xdf_dset = xdf_data[xdf_data["split"] == 'train'].copy()#.head(100)
xdf_dset_test = xdf_data[xdf_data["split"] == 'val'].copy()#.head(100)


train_questions = xdf_dset['question'].values
train_answers = xdf_dset['answer'].values

all_data = train_questions +train_answers
# Create a tokenizer
tokenizer = Tokenizer(BPE())
tokenizer.pre_tokenizer = Whitespace()

# Train the tokenizer on your data
trainer = BpeTrainer(min_frequency=2, special_tokens=["[SOS]","[EOS]","[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
tokenizer.train_from_iterator(all_data, trainer)

# Save the tokenizer
tokenizer.save(TOKENIZER_File)

# Encode questions and answers
# encoded_questions = tokenizer.encode_batch(train_questions)
# encoded_answers = tokenizer.encode_batch(train_answers)
# # Convert encoded sequences to lists of IDs
# question_ids = [encoding.ids for encoding in encoded_questions]
# answer_ids = [encoding.ids for encoding in encoded_answers]


def decoder_pair(sample = train_questions[0]):
    encoded_question = tokenizer.encode(sample)
    # Get the token IDs
    token_ids = encoded_question.ids
    # Get the attention mask
    attention_mask = encoded_question.attention_mask
    # Decode the token IDs back to words
    decoded_question = tokenizer.decode(token_ids)
    print("Token IDs:", token_ids)
    print("Attention Mask:", attention_mask)
    print("Decoded Question:", decoded_question)