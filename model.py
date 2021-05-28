from transformers import BertForTokenClassification, BertPreTrainedModel, BertConfig, BertTokenizer, BertModel, BertTokenizerFast, AutoModelForTokenClassification
from transformers import BertForSequenceClassification, AlbertTokenizer, AlbertForTokenClassification
from transformers import ElectraTokenizer, ElectraForTokenClassification, ElectraConfig, AutoTokenizer, ElectraModel
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers.modeling_outputs import TokenClassifierOutput
import torch
import torch.nn as nn
import numpy as np
from dataProcess import MAX_LEN, text_preprocessing
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm

BERT_MODEL_NAME =  "bert-base-chinese"#"hfl/chinese-bert-wwm-ext"
ALBERT_MODEL_NAME = "ckiplab/albert-base-chinese"
ELECTRA_MODEL_NAME = "hfl/chinese-electra-180g-base-discriminator"
RoBERTa_MODEL_NAME = "hfl/chinese-roberta-wwm-ext"
NUM_LABELS = 2
bert_tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
electra_tokenizer = ElectraTokenizer.from_pretrained(ELECTRA_MODEL_NAME)#AutoTokenizer.from_pretrained(ELECTRA_MODEL_NAME)
albert_tokenizer = BertTokenizerFast.from_pretrained(BERT_MODEL_NAME)
RoBERTa_tokenizer = BertTokenizer.from_pretrained(RoBERTa_MODEL_NAME)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#各種BERT模型
class BertClassifierGRU(nn.Module):
    """Bert Model for Classification Tasks.
    """
    def __init__(self, model_name, num_labels):
        super(BertClassifierGRU, self).__init__()
        # Specify hidden size of BERT, hidden size of our classifier, and number of labels
        
        # Instantiate BERT model
        self.num_labels = num_labels
        self.bert = BertModel.from_pretrained(model_name)
        self.bert_hidden_size = self.bert.config.hidden_size
        self.hidden_size = 64
        self.GRU1 = nn.GRU(self.bert_hidden_size, self.bert_hidden_size, bidirectional=True, batch_first=True)
        #self.hidden1 = nn.Linear(self.bert_hidden_size*2, self.hidden_size)
        self.clf = nn.Linear(self.bert_hidden_size*2, self.num_labels)
        self.relu = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, input_ids, labels):       
        
        # Feed input to BERT
        output = self.bert(input_ids=input_ids)
        outputs, _ = self.GRU1(output[0])
        outputs = self.dropout(outputs)
        #outputs = self.hidden1(outputs)
        #outputs = self.relu(outputs)
        logits = self.clf(outputs)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))               
        
        
        return TokenClassifierOutput(
            loss=loss,
            logits=logits
        )

class BertClassifierMultiGRU(nn.Module):
    """Bert Model for Classification Tasks.
    """
    def __init__(self, model_name, num_labels):
        super(BertClassifierMultiGRU, self).__init__()
        # Specify hidden size of BERT, hidden size of our classifier, and number of labels
        
        # Instantiate BERT model
        self.num_labels = num_labels
        self.bert = BertModel.from_pretrained(model_name)
        self.bert_hidden_size = self.bert.config.hidden_size
        self.hidden_size = 64
        self.GRU1 = nn.GRU(self.bert_hidden_size, self.bert_hidden_size, bidirectional=True, batch_first=True)
        self.GRU2 = nn.GRU(self.bert_hidden_size*2, self.bert_hidden_size, bidirectional=True, batch_first=True)
        #self.hidden1 = nn.Linear(self.bert_hidden_size*2, self.hidden_size)
        self.clf = nn.Linear(self.bert_hidden_size*2, self.num_labels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, input_ids, labels):       
        
        # Feed input to BERT
        output = self.bert(input_ids=input_ids)
        outputs, _ = self.GRU1(output[0])
        outputs = self.dropout(outputs)
        outputs, _ = self.GRU2(outputs)
        outputs = self.dropout(outputs)
        #outputs = self.hidden1(outputs)
        #outputs = self.relu(outputs)
        logits = self.clf(outputs)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))               
        
        
        return TokenClassifierOutput(
            loss=loss,
            logits=logits
        )

class BertClassifierLSTM(nn.Module):
    """Bert Model for Classification Tasks.
    """
    def __init__(self, model_name, num_labels):
        super(BertClassifierLSTM, self).__init__()
        # Specify hidden size of BERT, hidden size of our classifier, and number of labels
        
        # Instantiate BERT model
        self.num_labels = num_labels
        self.bert = BertModel.from_pretrained(model_name)
        self.bert_hidden_size = self.bert.config.hidden_size
        self.hidden_size = 64
        self.LSTM1 = nn.LSTM(self.bert_hidden_size, self.bert_hidden_size, bidirectional=True, batch_first=True)
        #self.hidden1 = nn.Linear(self.bert_hidden_size*2, self.hidden_size)
        self.clf = nn.Linear(self.bert_hidden_size*2, self.num_labels)
        self.relu = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, input_ids, labels):       
        
        # Feed input to BERT
        output = self.bert(input_ids=input_ids)
        outputs, _ = self.LSTM1(output[0])
        outputs = self.dropout(outputs)
        #outputs = self.hidden1(outputs)
        #outputs = self.relu(outputs)
        logits = self.clf(outputs)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))               
        
        
        return TokenClassifierOutput(
            loss=loss,
            logits=logits
        )

# Create the BertClassfier class
class BertSeqClassifier(nn.Module):
    """
    Bert Model for Classification Tasks.
    """
    def __init__(self, model_name, freeze_bert=False):
        super(BertSeqClassifier, self).__init__()
        # Specify hidden size of BERT, hidden size of our classifier, and number of labels
        D_in, H, D_out = 768, 50, 2

        # Instantiate BERT model
        self.bert = BertModel.from_pretrained(model_name)

        # Instantiate an one-layer feed-forward classifier
        self.classifier = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            #nn.Dropout(0.5),
            nn.Linear(H, D_out)
        )

        # Freeze the BERT model
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
    def forward(self, input_ids, labels, attention_mask):
        # Feed input to BERT
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)
        
        # Extract the last hidden state of the token `[CLS]` for classification task
        last_hidden_state_cls = outputs[0][:, 0, :]

        # Feed input to classifier to compute logits
        logits = self.classifier(last_hidden_state_cls)
        
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits, labels)
        return TokenClassifierOutput(
            loss=loss,
            logits=logits
        )

class ResBertClassifier(nn.Module):
    """
    Bert Model with residual learning for Classification Tasks.
    """
    def __init__(self, model_name, num_labels):
        super(ResBertClassifier, self).__init__()
        # Specify hidden size of BERT, hidden size of our classifier, and number of labels
        
        # Instantiate BERT model
        if 'elect' in model_name:
            print(model_name)
            self.tokenizer = ElectraTokenizer.from_pretrained(model_name)
            self.bert = ElectraModel.from_pretrained(model_name)
        else:
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
            self.bert = BertModel.from_pretrained(model_name)
        self.num_labels = num_labels
        self.bert_hidden_size = self.bert.config.hidden_size
        D_in, H, D_out = self.bert_hidden_size, 64, self.num_labels        
        #self.embedding = nn.Embedding(len(self.tokenizer.vocab), D_in)
        self.GRU1 = nn.GRU(D_in * 2, D_in, bidirectional=True, batch_first=True)
        # Instantiate an one-layer feed-forward classifier
        self.clf = nn.Sequential(
            nn.Linear(D_in * 2, H),
            #nn.ReLU(),
            #nn.Dropout(0.5),
            #nn.Linear(H, H),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(H, D_out)
        )
        
        self.hidden_size = 64
        
    def forward(self, input_ids, labels):       
        
        # Feed input to BERT
        inputs_embed = self.bert.embeddings(input_ids)
        output = self.bert(input_ids = input_ids, output_hidden_states = True)#
        
        output = torch.cat([output[0], torch.sub(output[0], inputs_embed)], -1)
        
        #output, _ = self.GRU1(output)
        logits = self.clf(output)
        #logits = self.clf(output[0])
                
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))     
        
        return TokenClassifierOutput(
            loss=loss,
            logits=logits
        )

class SepLayerBertClassifier(nn.Module):
    """
    Bert Model with residual learning for Classification Tasks.
    """
    def __init__(self, model_name, num_labels):
        super(SepLayerBertClassifier, self).__init__()
        # Specify hidden size of BERT, hidden size of our classifier, and number of labels
        
        # Instantiate BERT model
        if 'elect' in model_name:
            print(model_name)
            self.tokenizer = ElectraTokenizer.from_pretrained(model_name)
            self.bert = ElectraModel.from_pretrained(model_name)
        else:
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
            self.bert = BertModel.from_pretrained(model_name)
        self.num_labels = num_labels
        self.bert_hidden_size = self.bert.config.hidden_size
        D_in, H, D_out = self.bert_hidden_size, 64, self.num_labels        
        self.clf = nn.Sequential(
            nn.Linear(D_in, H),
            #nn.ReLU(),
            #nn.Dropout(0.5),
            #nn.Linear(H, H),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(H, D_out)
        )
        
        self.hidden_size = 64
        
    def forward(self, input_ids, labels):       
        
        # Feed input to BERT
        #inputs_embed = self.bert.embeddings(input_ids)
        output = self.bert(input_ids = input_ids, output_hidden_states = True)#
        
        logits = self.clf(sum(output[2]))
        #logits = self.clf(output[0])
                
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))     
        
        return TokenClassifierOutput(
            loss=loss,
            logits=logits
        )

#將輸入句轉換成BERT Token
def preprocessing_for_bert(data, module_type = 'TokenBERT'):
    # Create empty lists to store outputs
    input_ids = []
    if 'Seq' in module_type:
        attention_masks = []
        for sent in data:
            encoded_sent = bert_tokenizer.encode_plus(
                text=text_preprocessing(sent),  # Preprocess sentence
                add_special_tokens=True,        # Add `[CLS]` and `[SEP]`
                max_length=MAX_LEN,                  # Max length to truncate/pad
                pad_to_max_length=True,         # Pad sentence to max length
                #return_tensors='pt',           # Return PyTorch tensor
                return_attention_mask=True      # Return attention mask
                )
            
            # Add the outputs to the lists
            input_ids.append(encoded_sent.get('input_ids'))
            attention_masks.append(encoded_sent.get('attention_mask'))

        # Convert lists to tensors
        input_ids = torch.tensor(input_ids)
        attention_masks = torch.tensor(attention_masks)
        return input_ids, attention_masks
    elif 'ELEC' in module_type:
        print('ELECTRA')
        # For every sentence...
        for sent in data:
            encoded_sent = electra_tokenizer.encode_plus(
                text=text_preprocessing(sent),  # Preprocess sentence
                add_special_tokens=False,
                max_length=MAX_LEN,             # Max length to truncate/pad
                pad_to_max_length=True,         # Pad sentence to max length
                )
            
            # Add the outputs to the lists
            input_ids.append(encoded_sent.get('input_ids'))

        # Convert lists to tensors
        input_ids = torch.tensor(input_ids)

        return input_ids
    elif 'RoBERTa' in module_type:
        print('RoBERTa')
        # For every sentence...
        for sent in data:
            encoded_sent = RoBERTa_tokenizer.encode_plus(
                text=text_preprocessing(sent),  # Preprocess sentence
                add_special_tokens=False,
                max_length=MAX_LEN,             # Max length to truncate/pad
                pad_to_max_length=True,         # Pad sentence to max length
                )
            
            # Add the outputs to the lists
            input_ids.append(encoded_sent.get('input_ids'))

        # Convert lists to tensors
        input_ids = torch.tensor(input_ids)

        return input_ids    
    else:
        # For every sentence...
        for sent in data:
            encoded_sent = bert_tokenizer.encode_plus(
                text=text_preprocessing(sent),  # Preprocess sentence
                add_special_tokens=False,
                max_length=MAX_LEN,             # Max length to truncate/pad
                pad_to_max_length=True,         # Pad sentence to max length
                )
            
            # Add the outputs to the lists
            input_ids.append(encoded_sent.get('input_ids'))

        # Convert lists to tensors
        input_ids = torch.tensor(input_ids)

        return input_ids

#將dataloader
def get_dataloader(train_inputs, y_train, val_inputs, y_val, batch_size):
    # Convert other data types to torch.Tensor
    train_labels = torch.tensor(y_train)
    val_labels = torch.tensor(y_val)

    # Create the DataLoader for our training set
    train_data = TensorDataset(train_inputs, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    # Create the DataLoader for our validation set
    val_data = TensorDataset(val_inputs, val_labels)
    val_sampler = SequentialSampler(val_data)
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)

    return train_dataloader, val_dataloader

#初始化BERT模型
def initialize_model(module_type='normal', train_dataloader = [], epochs=4, isGPU = True):
    """
    Initialize the Bert Classifier, the optimizer and the learning rate scheduler.
    """
    # Instantiate Bert Classifier
    
    if module_type == 'LSTM':
        model = BertClassifierLSTM(BERT_MODEL_NAME, num_labels=NUM_LABELS)
    elif module_type == 'GRU':
        model = BertClassifierGRU(BERT_MODEL_NAME, num_labels=NUM_LABELS)
    elif module_type == 'MultiGRU':
        model = BertClassifierMultiGRU(BERT_MODEL_NAME, num_labels=NUM_LABELS)
    elif module_type == 'BERTForSeq':
        model = BertForSequenceClassification(BERT_MODEL_NAME)
    elif module_type == 'SeqBERT':
        model = BertSeqClassifier(BERT_MODEL_NAME)
    elif module_type == 'ELECTRA':
        model = ElectraForTokenClassification.from_pretrained(ELECTRA_MODEL_NAME, num_labels=NUM_LABELS)#AutoModelForTokenClassification.from_pretrained(ELECTRA_MODEL_NAME, num_labels=NUM_LABELS)
    elif module_type == 'RoBERTa':
        model = BertForTokenClassification.from_pretrained(RoBERTa_MODEL_NAME, num_labels=NUM_LABELS)
    elif 'Res' in module_type:  #ResBERT可選擇ELECTRA或RoBERTa進行訓練
        print(module_type)  
        if 'ELECT' in module_type:model = ResBertClassifier(ELECTRA_MODEL_NAME, num_labels=NUM_LABELS)
        else:model = ResBertClassifier(RoBERTa_MODEL_NAME, num_labels=NUM_LABELS)
    elif 'Sep' in module_type:  
        print(module_type)  
        if 'ELECT' in module_type:model = SepLayerBertClassifier(ELECTRA_MODEL_NAME, num_labels=NUM_LABELS)
        else:model = SepLayerBertClassifier(RoBERTa_MODEL_NAME, num_labels=NUM_LABELS)        
    elif module_type == 'ALBERT':
        model =  AutoModelForTokenClassification.from_pretrained(ALBERT_MODEL_NAME, num_labels=NUM_LABELS)    
    else:
        model = BertForTokenClassification.from_pretrained(BERT_MODEL_NAME, num_labels=NUM_LABELS)

    # Tell PyTorch to run the model on GPU
    if isGPU:
        model = model.to(device)

    # Create the optimizer
    optimizer = AdamW(model.parameters(),
                      lr=5e-5,    # Default learning rate
                      eps=1e-8    # Default epsilon value
                      )

    # Total number of training steps
    total_steps = len(train_dataloader) * epochs

    # Set up the learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0, # Default value
                                                num_training_steps=total_steps)
    return model, optimizer, scheduler

#讀取驗證檔
def get_test_data(test_path, testLabel='test_answer.txt', testFile='test_input.txt', batch_size=64):
    MAX_LEN = 140
    labels, texts = [], []
    with open(test_path+"\\"+testLabel, 'r',encoding='utf-8') as label:
        while True:
            line = label.readline().strip()
            if not line:break
            labels.append([int(i) for i in line]+[0 for j in range(MAX_LEN - len(line))])
        
    with open(test_path+"\\"+testFile,'r',encoding='utf-8') as text:    
        while True:
            line = text.readline().strip()
            if not line:break
            texts.append(line)
    
    #建立test dataloader
    test_inputs = preprocessing_for_bert(texts)
    test_labels = torch.tensor(labels)
    print(test_labels.size(), test_inputs.size())

    test_data = TensorDataset(test_inputs, test_labels)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)
    return test_dataloader

# 建立混淆矩陣
from sklearn.metrics import confusion_matrix
def show_confusion(y_real_s, y_pred_s):
    confmat = confusion_matrix(y_true=y_real_s, y_pred=y_pred_s)
    cm = {'tp': confmat[1, 1], 'fn': confmat[1, 0], 'fp': confmat[0, 1], 'tn': confmat[0, 0]}
    total = sum(cm.values())
    accuracy = (cm['tp']+cm['tn'])/total

    recall = (cm['tp'])/(cm['tp']+cm['fn'])

    precision = (cm['tp'])/(cm['tp']+cm['fp'])

    f1_score = (2*precision*recall)/(precision+recall)

    fa_rate = (cm['fp'])/(cm['tn']+cm['fp'])

    return accuracy, recall, precision, f1_score, fa_rate

def get_result(model, dataloader, module_type):
    MAX_LEN = 140
    y_real_s = []
    y_pred_s = []
    model.eval()
    for batch in dataloader:
        # Load batch to GPU
        b_input_ids, b_labels = tuple(t.to(device) for t in batch)    

        # Compute logits
        with torch.no_grad():
            logits = model(input_ids = b_input_ids, labels = b_labels)    
        # Compute loss
        loss = logits[0]

        # Get the predictions    

        for logit, real, b_input_id in zip(logits[1], b_labels, b_input_ids):
            pred = torch.max(logit, 1)[1].data #預測該字有錯的位置
                        
            corr = torch.zeros((1,MAX_LEN)).type(torch.int64)
            corr = corr.to(device)
            
            #建立混淆矩陣
            if torch.equal(real, corr[0]):y_real_s += [0] 
            else:y_real_s += [1]

            if torch.equal(pred, corr[0]):y_pred_s += [0]             
            else:y_pred_s += [1]

        #sentence = tokenizer.decode(b_input_ids)
    
    return show_confusion(y_real_s, y_pred_s)

def train(model, optimizer, scheduler, train_dataloader, ckptDir, eType, data_nums, epochs=4, isOld=False, finEpoch=20, module_type='normal'):
    Loss= []
    
    val_dataloader = get_test_data('data\\test', 'val_ans.txt', 'val_inp.txt', 128) #需修改
    model.train() #訓練模式
    if not isOld:
        for i in range(epochs):
            model.train()  
            batch_counts = 0
            running_loss = 0.0
            step_loss = 0.0
            for step, batch in enumerate(train_dataloader): #訓練資料
                batch_counts +=1
                # Load batch to GPU
                b_input_ids, b_labels = tuple(t.to(device) for t in batch)
                optimizer.zero_grad()
                out = model(input_ids=b_input_ids, labels=b_labels) #輸入資料
                loss = out[0] 
                loss.backward()  #將輸入更新模型
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) #梯度剪裁
                optimizer.step() #更新
                scheduler.step()
                running_loss += loss.item()
                '''
                step_loss += loss.item()
                if step % 200 == 0:                    
                    print('\r','[batch %2d] step_loss: %2.3f avg_step_loss: %.3f' % (step, step_loss, step_loss/200))
                    step_loss = 0.0
                '''
            #儲存模型
            torch.save(model.state_dict(), ckptDir+'\\'+'/M_{}_{}_E_{}_N_{}.h5'.format(module_type, eType, i+1, data_nums))
            Loss.append(running_loss)
            #print('[epoch %d] train_loss: %.3f avg_train_loss: %.3f' % (i + 1, running_loss, running_loss/len(train_dataloader)))
            
            # =======================================
            #               Evaluation
            # =======================================
            # on our validation set.
            
            #val_loss = evaluate(model, val_dataloader)
            if eType == 'WORD':
                model.eval()
                accuracy, recall, precision, f1_score, fa_rate = get_result(model, val_dataloader, module_type)
                print('\r','[epoch %2d] epoch_train_loss: %2.3f avg_train_loss: %2.3f val_acc:%2.3f val_f1:%2.3f val_recall:%2.3f val_pre:%2.3f ' % (i + 1, running_loss, running_loss/len(train_dataloader), accuracy, f1_score, recall, precision))
            else:
                print('\r','[epoch %2d] train_loss: %2.3f avg_train_loss: %.3f' % (i + 1, running_loss, running_loss/len(train_dataloader)))            
    
    return Loss

if __name__ == "__main__":
    from dataProcess import get_data, write_Wrong_word
    write_Wrong_word('data\\train', 'newsCorpus.txt', 'train_dataset.txt', 1e3)
    x_train, x_val, y_train, y_val = get_data('data\\train', 'train_dataset.txt')
    val_inputs = preprocessing_for_bert(x_val, module_type='ResRoBERTa')
    train_inputs = preprocessing_for_bert(x_train, module_type='ResRoBERTa')
    train_dataloader, val_dataloader = get_dataloader(train_inputs, y_train, val_inputs, y_val, 64)
    bert_classifier, optimizer, scheduler = initialize_model('SepRoBERTa', train_dataloader, 1, False)
    tmp_loader = iter(val_dataloader)
    tmp_ids, tmp_labels = next(tmp_loader)
    #print(tmp_ids, tmp_labels)
    bert_classifier(tmp_ids, tmp_labels)
    #print(bert_classifier(tmp_ids, tmp_labels))