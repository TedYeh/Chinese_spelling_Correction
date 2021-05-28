import pandas as pd
import random
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import torch
from model import device, bert_tokenizer, preprocessing_for_bert, initialize_model, electra_tokenizer, albert_tokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

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
    test_inputs = preprocessing_for_bert(texts)
    test_labels = torch.tensor(labels)
    print(test_labels.size(), test_inputs.size())

    test_data = TensorDataset(test_inputs, test_labels)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)
    return test_dataloader

def show_confusion(y_real_s, y_pred_s, y_real_c, y_pred_c):
    confmat_s = confusion_matrix(y_true=y_real_s, y_pred=y_pred_s)
    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    ax.matshow(confmat_s, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(confmat_s.shape[0]):
        for j in range(confmat_s.shape[1]):
            ax.text(x=j, y=i, s=confmat_s[i,j], va='center', ha='center')
    print(confmat_s[1, 0])
    cm_s = {'tp': confmat_s[1, 1], 'fn': confmat_s[1, 0], 'fp': confmat_s[0, 1], 'tn': confmat_s[0, 0]}
    total = sum(cm_s.values())
    print('--------------Detection level--------------')
    print("accuracy:", (cm_s['tp']+cm_s['tn'])/total)

    recall = (cm_s['tp'])/(cm_s['tp']+cm_s['fn'])
    print("recall:", recall)

    precision = (cm_s['tp'])/(cm_s['tp']+cm_s['fp'])
    print("precision:", precision)

    print("f1-score:", 2/((1/precision)+(1/recall)))

    fa_rate = (cm_s['fp'])/(cm_s['tn']+cm_s['fp'])
    print("FA-Rate:", fa_rate)
    plt.xlabel('Prediction')        
    plt.ylabel('Real Label')
    plt.show()

    confmat_c = confusion_matrix(y_true=y_real_c, y_pred=y_pred_c)
    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    ax.matshow(confmat_c, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(confmat_c.shape[0]):
        for j in range(confmat_c.shape[1]):
            ax.text(x=j, y=i, s=confmat_c[i,j], va='center', ha='center')

    cm_c = {'tp': confmat_c[1, 1], 'fn': confmat_c[1, 0], 'fp': confmat_c[0, 1], 'tn': confmat_c[0, 0]}
    total = sum(cm_c.values())
    print('--------------Position level--------------')
    print("accuracy:", (cm_c['tp']+cm_c['tn'])/total)

    recall = (cm_c['tp'])/(cm_c['tp']+cm_c['fn'])
    print("recall:", recall)

    precision = (cm_c['tp'])/(cm_c['tp']+cm_c['fp'])
    print("precision:", precision)

    print("f1-score:", 2/((1/precision)+(1/recall)))

    fa_rate = (cm_c['fp'])/(cm_c['tn']+cm_c['fp'])
    print("FA-Rate:", fa_rate)
    plt.xlabel('Prediction')        
    plt.ylabel('Real Label')
    plt.show()    

mode = "sentence1times_eWord"
def get_test_result(model, dataloader, num, etype, epochs, module_type, correction_model=None):
    model.eval()
    if correction_model:correction_model.eval()
    MAX_LEN = 140
    y_real_s, y_pred_s = [], []
    y_real_c, y_pred_c = [], []
    cor_count = 0
    
    data = {
    "真實標記":[],
    "預測標記":[]
    }
    
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
            token = [int(i) for i in b_input_id.cpu().numpy() if i != 0] 
            pred = torch.max(logit, 1)[1].data
            if 'ELECTRA' in module_type:sentence = electra_tokenizer.decode(b_input_id[:len(token)]) #轉成中文句
            elif module_type == 'ALBERT':sentence = albert_tokenizer.decode(b_input_id[:len(token)]) #轉成中文句
            else:sentence = bert_tokenizer.decode(b_input_id[:len(token)]) #轉成中文句
            realList = sentence.split()
            predList = sentence.split()
            predLabel = torch.nonzero(pred)
            realLabel = torch.nonzero(real)
            #predLabel2 = torch.nonzero(pred, as_tuple=True)
            #realLabel2 = torch.nonzero(real, as_tuple=True)
            #print(pred.cpu().numpy() == real.cpu().numpy())
            if 1 in real.cpu().numpy():
                for i in range(len(real.cpu().numpy())):
                    #建立混淆矩陣
                    if real.cpu().numpy()[i] == 1 and pred.cpu().numpy()[i] == 1:#TP
                        y_real_c += [1] 
                        y_pred_c += [1]
                    elif real.cpu().numpy()[i] != pred.cpu().numpy()[i]:#FN
                        y_real_c += [real.cpu().numpy()[i]]
                        y_pred_c += [pred.cpu().numpy()[i]]
            else:
                if torch.equal(pred, real):#TN
                    y_real_c += [0]
                    y_pred_c += [0]
                else:    
                    for i in range(len(real.cpu().numpy())):
                        if real.cpu().numpy()[i] != pred.cpu().numpy()[i]:#FP
                            y_real_c += [real.cpu().numpy()[i]]
                            y_pred_c += [pred.cpu().numpy()[i]]

            '''            
            for i in range(len(pred.cpu().numpy())):
                if real.cpu().numpy()[i] == 0:y_real_c += [0]
                else:y_real_c += [1]
                if pred.cpu().numpy()[i] == 0:y_pred_c += [0]
                else:y_pred_c += [1]
            '''

            if realLabel.size()[0] != 0:
                if realLabel[0].cpu().numpy() >= len(realList):continue
                if realLabel[-1].cpu().numpy() >= len(realList):continue

            if predLabel.size()[0] != 0:
                if predLabel[0].cpu().numpy() >= len(predList):continue
                if predLabel[-1].cpu().numpy() >= len(predList):continue  

            #標記有錯的別字
            if predLabel.size()[0] == 1: 
                predList[predLabel[0]] = '['+predList[predLabel[0]]+']'
            elif predLabel.size()[0] > 1:
                for i in predLabel:
                    predList[i[0]] = '['+predList[i[0]]+']'
            data["預測標記"].append(''.join(predList)) 

            if realLabel.size()[0] == 1:            
                realList[realLabel[0][0]] = '['+realList[realLabel[0][0]]+']'
            elif realLabel.size()[0] > 1:
                for i in realLabel:
                    realList[i[0]] = '['+realList[i[0]]+']'
            data["真實標記"].append(''.join(realList))

            corr = torch.zeros((1,MAX_LEN)).type(torch.int64)
            corr = corr.to(device)

            if torch.equal(real, corr[0]):y_real_s += [0]
            else:y_real_s += [1]

            if torch.equal(pred, corr[0]):y_pred_s += [0] 
            elif torch.equal(pred, real):
                y_pred_s += [1]
                cor_count += 1
            else:
                y_pred_s += [1]

        #sentence = tokenizer.decode(b_input_ids)
    df = pd.DataFrame(data)
    df.to_csv('result/T_{}_E_{}_N_{}.csv'.format(etype, epochs, num))
    print(cor_count)
    print(df)
    return y_real_s, y_pred_s, y_real_c, y_pred_c


