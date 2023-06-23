import torch
from torch.utils.data import Dataset

class GeneratorDataset(Dataset):

    def __init__(self, dataframe, tokenizer, source_len, summ_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = source_len
        self.summ_len = summ_len
        self.bugid = self.data.bugid
        self.buggy = self.data.buggy
        self.patch = self.data.patch
        self.bug = self.data.bug

    def __len__(self):
        return len(self.patch)

    def __getitem__(self, index):
        buggy = str(self.buggy[index])
        buggy = ' '.join(buggy.split())

        patch = str(self.patch[index])
        patch = ' '.join(patch.split())

        source = self.tokenizer.batch_encode_plus([buggy], max_length= self.source_len, pad_to_max_length=True,return_tensors='pt')
        target = self.tokenizer.batch_encode_plus([patch], max_length= self.summ_len, pad_to_max_length=True,return_tensors='pt')

        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        target_ids = target['input_ids'].squeeze()
        target_mask = target['attention_mask'].squeeze()

        return {
            'bugid': torch.tensor(self.bugid[index], dtype=torch.long),
            'source_ids': source_ids.to(dtype=torch.long), 
            'source_mask': source_mask.to(dtype=torch.long), 
            'target_ids': target_ids.to(dtype=torch.long),
            'target_ids_y': target_ids.to(dtype=torch.long),
            'bug': self.bug[index]
        }
    
class GeneratorDatasetForMultiSource(Dataset):
    def __init__(self, dataframe, tokenizer, source_len, summ_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = source_len
        self.summ_len = summ_len
        self.text_data_1 = self.data.additional_info
        self.text_data_2 = self.data.buggy
        self.labels = self.data.patch
        self.bugid = self.data.bugid
        self.bug = self.data.bug

    def __len__(self):
        return len(self.text_data_1)

    def __getitem__(self, index):
        text_1 = self.text_data_1[index]
        text_2 = self.text_data_2[index]
        label = self.labels[index]

        # Tokenize text inputs
        text_input_1 = self.tokenizer.batch_encode_plus([text_1], max_length= self.source_len, pad_to_max_length=True,return_tensors='pt', truncation=True)
        text_input_2 = self.tokenizer.batch_encode_plus([text_2], max_length= self.source_len, pad_to_max_length=True,return_tensors='pt', truncation=True)
        target = self.tokenizer.batch_encode_plus([label], max_length= self.summ_len, pad_to_max_length=True,return_tensors='pt', truncation=True)


        input_ids_1 = text_input_1['input_ids'].squeeze()
        attention_mask_1 = text_input_1['attention_mask'].squeeze()
        input_ids_2 = text_input_2['input_ids'].squeeze()
        attention_mask_2 = text_input_2['attention_mask'].squeeze()

        # print(input_ids_1, input_ids_2)

        target_ids = target['input_ids'].squeeze()
        target_mask = target['attention_mask'].squeeze()

        # input_ids = torch.cat((input_ids_1, input_ids_2), dim=1)
        # attention_mask = torch.cat((attention_mask_1, attention_mask_2), dim=1)

        return {
            'input_ids_1': input_ids_1.to(dtype=torch.long), 
            'attention_mask_1': attention_mask_1.to(dtype=torch.long), 
            'input_ids_2': input_ids_2.to(dtype=torch.long), 
            'attention_mask_2': attention_mask_2.to(dtype=torch.long), 
            'target_ids': target_ids.to(dtype=torch.long),
            'target_ids_y': target_ids.to(dtype=torch.long),
            'bugid': torch.tensor(self.bugid[index], dtype=torch.long),
            'bug': self.bug[index]
        }


class CustomDataset(Dataset):

    def __init__(self, dataframe, tokenizer, source_len, summ_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = source_len
        self.summ_len = summ_len
        self.buggy = self.data.buggy
        self.patch = self.data.patch

    def __len__(self):
        return len(self.patch)

    def __getitem__(self, index):
        buggy = str(self.buggy[index])
        buggy = ' '.join(buggy.split())

        patch = str(self.patch[index])
        patch = ' '.join(patch.split())

        source = self.tokenizer.batch_encode_plus([buggy], max_length= self.source_len,pad_to_max_length=True,return_tensors='pt')
        target = self.tokenizer.batch_encode_plus([patch], max_length= self.summ_len, pad_to_max_length=True,return_tensors='pt')

        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        target_ids = target['input_ids'].squeeze()
        target_mask = target['attention_mask'].squeeze()

        return {
            'source_ids': source_ids.to(dtype=torch.long), 
            'source_mask': source_mask.to(dtype=torch.long), 
            'target_ids': target_ids.to(dtype=torch.long),
            'target_ids_y': target_ids.to(dtype=torch.long)
        }