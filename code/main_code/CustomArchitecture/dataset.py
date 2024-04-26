import torch
import torch.nn as nn
from torch.utils.data import Dataset

from PIL import Image
import torchvision.transforms as transforms

from transformers import  BlipForQuestionAnswering,BlipProcessor
blipprocessor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")

#blip_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
class ImageProcessor(nn.Module):
    def __init__(self):
        super().__init__()
        #self.blip_vision_model = blip_vision_model#blip_model.vision_model
        self.transform = transforms.Compose([
            transforms.Resize((384, 384)),  # Resize the image to the desired size
            transforms.ToTensor(),  # Convert the image to a tensor
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize the image
        ])

    def forward(self, image_path):
        # Load and preprocess the image
        #image = self.load_and_preprocess_image(image_path)
        # Pass the preprocessed image through the BLIP vision model to generate embeddings

        #image_embeddings = self.blip_vision_model(image)
        #image_embeddings['last_hidden_state']

        #using processor
        image = Image.open(image_path).convert('RGB')
        encoding = blipprocessor(image,return_tensors="pt")
        return encoding

    def load_and_preprocess_image(self, image_path):
        # Open the image file
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        image = image.unsqueeze(0)
        return image


class QuestionAnswerDataset(Dataset):
    def __init__(self, ds, tokenizer,ans_seq_len,qstn_seq_len):# seq_len):
        super().__init__()
        #self.seq_len = seq_len
        self.ans_seq_len = ans_seq_len
        self.qstn_seq_len = qstn_seq_len
        self.ds = ds
        self.tokenizer = tokenizer
        self.image_processor = ImageProcessor()
        self.sos_token = torch.tensor([tokenizer.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        try:
            question_answer_pair = self.ds.loc[idx]
            question_text = question_answer_pair['question']
            answer_text = question_answer_pair['answer']

            pixel_values = self.image_processor.forward(question_answer_pair['image_path'])['pixel_values']
            #image_embeddings = ImageProcessor(question_answer_pair['image_path'])
        except KeyError:
            raise IndexError(f"Index {idx} is out of bounds for dataset of size {len(self)}")

        # question_answer_pair = self.ds.loc[idx]
        # print(f"Error : {idx} \n {self.ds.loc[idx]}")
        # question_text = question_answer_pair['question']
        # answer_text = question_answer_pair['answer']

        # Transform the text into tokens
        question_tokens = self.tokenizer.encode(question_text).ids
        answer_tokens = self.tokenizer.encode(str(answer_text)).ids

        # Add sos, eos and padding to each sentence
        # question_num_padding_tokens = self.seq_len - len(question_tokens) - 2  # We will add <s> and </s>
        # answer_num_padding_tokens = self.seq_len - len(answer_tokens) - 1  # We will only add <s>, and </s> only on the label
        question_num_padding_tokens = self.qstn_seq_len - len(question_tokens) - 2  # We will add <s> and </s>
        answer_num_padding_tokens = self.ans_seq_len - len(answer_tokens) - 1  # We will only add <s>, and </s> only on the label

        # Make sure the number of padding tokens is not negative. If it is, the sentence is too long
        if question_num_padding_tokens < 0 or answer_num_padding_tokens < 0:
            print(question_text)
            raise ValueError("Sentence is too long")

        # Add <s> and </s> token
        question_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(question_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * question_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Add only <s> token
        answer_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(answer_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * answer_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Add only </s> token
        label = torch.cat(
            [
                torch.tensor(answer_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * answer_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Double check the size of the tensors to make sure they are all seq_len long
        assert question_input.size(0) == self.qstn_seq_len#self.seq_len
        assert answer_input.size(0) == self.ans_seq_len#self.seq_len
        assert label.size(0) == self.ans_seq_len#self.seq_len

        return {
            "question_input": question_input,  # (seq_len)
            "answer_input": answer_input,  # (seq_len)
            "question_mask": (question_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),  # (1, 1, seq_len)
            "answer_mask": (answer_input != self.pad_token).unsqueeze(0).int() & causal_mask(answer_input.size(0)),  # (1, seq_len) & (1, seq_len, seq_len)
            "label": label,  # (seq_len)
            "question_text": question_text,
            "answer_text": answer_text,
            "pixel_values" : pixel_values
        }

def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0