import torch
import torch.nn as nn
from torch.utils.data import Dataset

class QuestionAnswerDataset(Dataset):
    def __init__(self, ds, tokenizer, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.ds = ds
        self.tokenizer = tokenizer
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

        except KeyError:
            raise IndexError(f"Index {idx} is out of bounds for dataset of size {len(self)}")

        # question_answer_pair = self.ds.loc[idx]
        # print(f"Error : {idx} \n {self.ds.loc[idx]}")
        # question_text = question_answer_pair['question']
        # answer_text = question_answer_pair['answer']

        # Transform the text into tokens
        question_tokens = self.tokenizer.encode(question_text).ids
        answer_tokens = self.tokenizer.encode(answer_text).ids

        # Add sos, eos and padding to each sentence
        question_num_padding_tokens = self.seq_len - len(question_tokens) - 2  # We will add <s> and </s>
        answer_num_padding_tokens = self.seq_len - len(answer_tokens) - 1  # We will only add <s>, and </s> only on the label

        # Make sure the number of padding tokens is not negative. If it is, the sentence is too long
        if question_num_padding_tokens < 0 or answer_num_padding_tokens < 0:
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
        assert question_input.size(0) == self.seq_len
        assert answer_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "question_input": question_input,  # (seq_len)
            "answer_input": answer_input,  # (seq_len)
            "question_mask": (question_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),  # (1, 1, seq_len)
            "answer_mask": (answer_input != self.pad_token).unsqueeze(0).int() & causal_mask(answer_input.size(0)),  # (1, seq_len) & (1, seq_len, seq_len)
            "label": label,  # (seq_len)
            "question_text": question_text,
            "answer_text": answer_text,
        }

def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0