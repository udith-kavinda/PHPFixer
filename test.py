# Importing stock libraries
import numpy as np
import pandas as pd
import torch, csv
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

# Importing the T5 modules from huggingface/transformers
from transformers import T5Tokenizer
# # Setting up the device for GPU usage
from torch import cuda
import gc
import warnings
import loader
from model_source.t5_for_multi_source import T5ForMultiSourceConditionalGeneration

        
def test(epoch, tokenizer, model, device, loader):
    return_sequences = 100
    model.eval()

    with torch.no_grad():
        for _, data in enumerate(loader, 0):
            gc.collect()
            torch.cuda.empty_cache()
            y = data['target_ids'].to(device, dtype = torch.long)
            input_ids_1 = data['input_ids_1'].to(device, dtype = torch.long)
            attention_mask_1 = data['attention_mask_1'].to(device, dtype = torch.long)
            input_ids_2 = data['input_ids_1'].to(device, dtype = torch.long)
            attention_mask_2 = data['attention_mask_1'].to(device, dtype = torch.long)
            bugid = data['bugid'].to(device, dtype = torch.long)

            input_ids = torch.cat((input_ids_1, input_ids_2), dim=1)
            attention_mask = torch.cat((attention_mask_1, attention_mask_2), dim=1)
            
            if _%10==0:
                print(_)
       
            generated_ids = model.generate(
                input_ids = input_ids,
                attention_mask = attention_mask, 
                max_length=100, 
                num_beams=return_sequences,
                length_penalty=1.0, 
                early_stopping = True,
                num_return_sequences=return_sequences,
                num_beam_groups = 1,
                output_scores=True
                )
           

            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]

            
            with open('result.csv', 'a') as csvfile:
                filewriter = csv.writer(csvfile, delimiter='\t',escapechar=' ',quoting=csv.QUOTE_NONE)
                for i in range(0,return_sequences):
                    predstr=preds[i]
                    predstr=predstr.replace('> =','>=').replace('< =','<=').replace('= =','==').replace('! =','!=')
                    filewriter.writerow([bugid.item(), data['bug'][0] , predstr])






def main():
     
    TRAIN_BATCH_SIZE =20    # input batch size for training (default: 64)
    VAL_EPOCHS = 1 
    LEARNING_RATE = 1e-4    # learning rate (default: 0.01)
    SEED = 0               # random seed (default: 42)
    MAX_LEN = 512
    SUMMARY_LEN = 512 
    SAVE_MODEL='./model/t5-base-serial'

    # Set random seeds and deterministic pytorch for reproducibility
    torch.manual_seed(SEED) # pytorch random seed
    np.random.seed(SEED) # numpy random seed
    torch.backends.cudnn.deterministic = True
    torch.cuda.empty_cache()

    # tokenzier for encoding the text
    tokenizer = T5Tokenizer.from_pretrained(SAVE_MODEL)
    # tokenizer.add_tokens(['{', '}','<','^'])

    device = 'cuda' if cuda.is_available() else 'cpu'
    model = T5ForMultiSourceConditionalGeneration.from_pretrained(SAVE_MODEL).to(device)
    # Further this model is sent to device (GPU/TPU) for using the hardware.


    test_df = pd.read_csv('./data/test.csv',encoding='latin-1',delimiter='\t')
    print(test_df.head())
    test_df = test_df[['bugid', 'bug','buggy', 'additional_info','patch']]
    print(test_df.head())

    
    test_dataset=test_df.reset_index(drop=True)


    print("TEST Dataset: {}".format(test_dataset.shape))



    test_set = loader.GeneratorDatasetForMultiSource(test_dataset, tokenizer, MAX_LEN, SUMMARY_LEN)

    
    test_params = {
        'batch_size': 1,
        'shuffle': False,
        'num_workers': 2
        }

    test_loader = DataLoader(test_set, **test_params)  

    # Defining the optimizer that will be used to tune the weights of the network in the training session. 
    optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)

    for epoch in range(0,1):
   
        test(epoch, tokenizer, model, device, test_loader)
        
        
        
if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    device = 'cuda' if cuda.is_available() else 'cpu'
    print(torch.__version__)
    gc.collect()
    torch.cuda.empty_cache()
    main()

