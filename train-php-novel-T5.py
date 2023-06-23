import numpy as np
import pandas as pd
import torch, csv
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer
from torch import cuda
import gc
import warnings
import loader
import BugsPHPDiscriminator
import torch.autograd as autograd
from model_source.t5_for_multi_source import T5ForMultiSourceConditionalGeneration


class CustomDataset(Dataset):

    def __init__(self, dataframe, tokenizer, source_len, summ_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = source_len
        self.summ_len = summ_len
        self.text_data_1 = self.data.additional_info
        self.text_data_2 = self.data.buggy
        self.labels = self.data.patch

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
            'target_ids_y': target_ids.to(dtype=torch.long)
        }

class GeneratorDatasetForMultiSource(Dataset):
    def __init__(self, dataframe, tokenizer, source_len, summ_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = source_len
        self.summ_len = summ_len
        self.text_data_1 = self.data.buggy
        self.text_data_2 = self.data.additional_info
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

def semantic_training(generator, gen_opt, gen_tokenizer, adv_loader, device,epoch):

    generator.train()
    
    for _,data in enumerate(adv_loader, 0):
        y = data['target_ids'].to(device, dtype = torch.long)
        y_ids = y[:, :-1].contiguous()
        lm_labels = y[:, 1:].clone().detach()
        lm_labels[y[:, 1:] == gen_tokenizer.pad_token_id] = -100

        input_ids_1 = data['input_ids_1'].to(device, dtype = torch.long)
        attention_mask_1 = data['attention_mask_1'].to(device, dtype = torch.long)
        input_ids_2 = data['input_ids_2'].to(device, dtype = torch.long)
        attention_mask_2 = data['attention_mask_2'].to(device, dtype = torch.long)

        input_ids = torch.cat((input_ids_1, input_ids_2), dim=1)
        attention_mask = torch.cat((attention_mask_1, attention_mask_2), dim=1)
        
        # ids = data['source_ids'].to(device, dtype = torch.long)
        # mask = data['source_mask'].to(device, dtype = torch.long)
        bugid = data['bugid'].to(device, dtype = torch.long)
        bug = data['bug']
        # print(f'bugid: {bugid}')
        
                
        bugcode = input_ids_2[0]
        end_index=getEndIndex(bugcode,32108) #2625 is the index for 'context',32108 is the index of 'context:'       
        bugcode = bugcode[3:end_index-1] #your index may be different!
        buggy = [gen_tokenizer.decode(bugcode, skip_special_tokens=True, clean_up_tokenization_spaces=True)]
           
            
        outputs = generator(input_ids = input_ids, attention_mask = attention_mask, decoder_input_ids=y_ids, labels=lm_labels)
        loss = outputs[0]
        # print(f'original loss: {loss}')
        lm_logits = outputs[1]
        output = F.log_softmax(lm_logits, -1)
        preds_seq = output.max(2)[1]

        g = preds_seq[0]
        end_index=getEndIndex(g,1)            
        g = g[:end_index]      



        preds = [gen_tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True)]
        predstr = preds[0] 
        print(f'predstr: {predstr}')

        # identity discriminator
        identity_reward = identity_discriminator(buggy[0], predstr)

        reward = autograd.Variable(torch.FloatTensor([1.0]))

        if 'same' in identity_reward:
            reward = autograd.Variable(torch.FloatTensor([1.4]))
        else:            
            reward = validate_by_compiler(bugid, predstr, bug)
        
        print(f'reward: {reward}')
  
        #combine cross entropy loss and compiler reward loss
        reward = reward.to(device)   
        loss = outputs[0]*reward
        print(f'semantic loss: {loss}')

        gen_opt.zero_grad()
        loss.backward()
        gen_opt.step()        

        recordData(epoch, bugid.item(), outputs[0].item(), reward.item(), predstr )             

def recordData(epoch, bugid, crossEntropLoss, reward, preds):
    with open('./logs.csv', 'a') as csvfile:
        filewriter = csv.writer(csvfile, delimiter='\t',quotechar='"',quoting=csv.QUOTE_MINIMAL)
        filewriter.writerow([epoch, bugid, crossEntropLoss, reward, preds])   

                
def getEndIndex(g,index):
    end_index=0
    for i in g:
        end_index+=1
        # 1 for </s>
        if i == index:
            break
    return end_index
                
                
def identity_discriminator(buggy, predstr):
    print(f'buggy: {buggy}')
    print(f'predstr: {predstr}')
    if buggy in predstr and predstr in buggy:
        return 'same'
    else:
        return 'different'
      
       
    
def validate_by_compiler(bugid, preds, bug):
    R = 0.2
    result = BugsPHPDiscriminator.getResults(bugid.item(), preds, rootPath, bug)
    print(f'result: {result}')
    if 'noTestResults' in result:
        rewardValue=1+R
    elif 'failedFailingTests' in result:
        rewardValue=1-R
    elif 'passedFailingTests' in result:
        rewardValue=1-R*2
    elif 'passAllTests' in result:
        rewardValue=1-R*3
    else:
        rewardValue=1
        
    
    return autograd.Variable(torch.FloatTensor([rewardValue]))

    

def syntrain(epoch, tokenizer, model, device, loader, optimizer):
    model.train()
    countInt = 0
    print(len(loader))
    for idx,data in enumerate(loader, 0):
  
        y = data['target_ids'].to(device, dtype = torch.long)
        y_ids = y[:, :-1].contiguous()
        lm_labels = y[:, 1:].clone().detach()
        lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100

        input_ids_1 = data['input_ids_1'].to(device, dtype = torch.long)
        attention_mask_1 = data['attention_mask_1'].to(device, dtype = torch.long)
        input_ids_2 = data['input_ids_2'].to(device, dtype = torch.long)
        attention_mask_2 = data['attention_mask_2'].to(device, dtype = torch.long)

        input_ids = torch.cat((input_ids_1, input_ids_2), dim=1)
        attention_mask = torch.cat((attention_mask_1, attention_mask_2), dim=1)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=y_ids, labels=lm_labels)
    
        loss = outputs[0]
        loss.backward()
        optimizer.step()


        if idx%1000 ==0:
            print(f'Syntatic Train Epoch: {epoch}, Loss:  {loss.item()}')
            print(idx)

        # we also save the model here in case of an accident during training
        if idx%10000 ==0:
            model.save_pretrained(SAVE_MODEL)
            tokenizer.save_pretrained(SAVE_MODEL)

        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
        
        
        
def valid( tokenizer, model, device, loader, optimizer):
    model.eval()
    total_loss = 0 
    total_nb=0
    with torch.no_grad():
        for _,data in enumerate(loader, 0):
            y = data['target_ids'].to(device, dtype = torch.long)
            y_ids = y[:, :-1].contiguous()
            lm_labels = y[:, 1:].clone().detach()
            lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
            ids = data['source_ids'].to(device, dtype = torch.long)
            mask = data['source_mask'].to(device, dtype = torch.long)

            outputs = model(input_ids = ids, attention_mask = mask, decoder_input_ids=y_ids, labels=lm_labels)
            loss = outputs[0]
            total_nb += 1  
            total_loss += loss.item()    

        print(f'Total Loss:  {total_loss}/{total_nb}')
        


def getGeneratorDataLoader(filepatch,tokenizer,batchsize):
    df = pd.read_csv(filepatch,encoding='latin-1',delimiter='\t')
    print(df.head(1))
    
    df = df[['bugid','bug', 'buggy', 'additional_info','patch']]

    params = {
        'batch_size': batchsize,
        'shuffle': False,
        'num_workers': 0
        }

    dataset=df.reset_index(drop=True)
    target_set = GeneratorDatasetForMultiSource(dataset, tokenizer, MAX_LEN, PATCH_LEN)
    target_loader = DataLoader(target_set, **params)
    return target_loader
        



def syntactic(epoch,syn_train_data_path):  
    # Set random seeds and deterministic pytorch for reproducibility
    torch.manual_seed(SEED) # pytorch random seed
    np.random.seed(SEED) # numpy random seed
    torch.backends.cudnn.deterministic = True
    torch.cuda.empty_cache()
    
    # Process data
    df = pd.read_csv(syn_train_data_path,encoding='latin-1',delimiter='\t', header=0, error_bad_lines=False, nrows=100).dropna()
    print(df.head())
    df = df[['bugid','buggy','additional_info','patch']]
    print(df.head())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = T5ForMultiSourceConditionalGeneration.from_pretrained(SAVE_MODEL, output_hidden_states=True).to(device)
    tokenizer = T5Tokenizer.from_pretrained(SAVE_MODEL, truncation=True)

    # tokenzier for encoding the text
    if epoch == 0 and 'pretrain' in syn_train_data_path:
        tokenizer.add_tokens(['{', '}','<','>','^','>=','<=','=','!=','==','!==','===','$','->','::',':','<?php',
                      'string', 'float', 'integer', 'boolean', 'array', 'unknown', 'buggy:','context:','type_info:','global_variable:','function_name:'])
        model.resize_token_embeddings(len(tokenizer))
    
    # Creation of Dataset and Dataloader
    train_dataset=df.reset_index(drop=True)     
    print("TRAIN Dataset: {}".format(train_dataset.shape))

    # Creating the Training and Validation dataset for further creation of Dataloader
    training_set = CustomDataset(train_dataset, tokenizer, MAX_LEN, PATCH_LEN)

    # Defining the parameters for creation of dataloaders
    train_params = {
        'batch_size': TRAIN_BATCH_SIZE,
        'shuffle': True,
        'num_workers': 2
        }    

    # # Creation of Dataloaders for testing and validation. 
    training_loader = DataLoader(training_set, **train_params)
  

    # Defining the optimizer that will be used to tune the weights of the network in the training session. 
    optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE) 
            
    syntrain(epoch, tokenizer, model, device, training_loader, optimizer)
    model.save_pretrained(SAVE_MODEL)
    tokenizer.save_pretrained(SAVE_MODEL)
    print(f'Syntatic Train Model Saved: {epoch}')

        
        
def semantic(epoch):   
    
    gen = T5ForMultiSourceConditionalGeneration.from_pretrained(SAVE_MODEL, output_hidden_states=True)    
    gen_tokenizer = T5Tokenizer.from_pretrained(SAVE_MODEL,truncation=True)
    gen = gen.to(device)   
    gen_optimizer = torch.optim.Adam(params = gen.parameters(), lr=LEARNING_RATE)
    data_loader=getGeneratorDataLoader(semantic_train_data_path,gen_tokenizer,1)   

    print('\n---Semantic Training-----\nEPOCH %d\n--------' % (epoch+1))

    # train model
    semantic_training(gen, gen_optimizer, gen_tokenizer, data_loader, device, epoch)                       
    # save trained model       
    gen.save_pretrained(SAVE_MODEL)
    gen_tokenizer.save_pretrained(SAVE_MODEL)  
    print(f'Sementic Train Model Saved: {epoch}')

                
        
        
if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    device = 'cuda' if cuda.is_available() else 'cpu'
    print(torch.__version__)
    gc.collect()
    torch.cuda.empty_cache()

    syn_train_data_path_1= './data/pretrain.csv'
    semantic_train_data_path = 'BugsPHP_Training/test.csv'
    SAVE_MODEL='./model/t5-base-serial'
    rootPath='/your/path/'
    TRAIN_BATCH_SIZE = 15   
    TRAIN_EPOCHS = 15      # number of epochs to train 
    LEARNING_RATE = 1e-4    # learning rate
    SEED = 42               # random seed (default: 42)
    MAX_LEN = 512
    PATCH_LEN = 100    

    
    #we train the syntactic training and semantic training
    for epoch in range(0,TRAIN_EPOCHS):
        syntactic(epoch,syn_train_data_path_1)
        # if  (epoch> 5 and epoch % 3 == 0) or epoch == TRAIN_EPOCHS-1:
        #     semantic(epoch)
