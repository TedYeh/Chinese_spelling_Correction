import re
from sklearn.model_selection import train_test_split
from proCorpus import preProcess
import numpy as np

MAX_LEN = 140
def get_data(path, name, test_size = 0.0075):
    x, y = [], []
    with open(path+"\\"+name, 'r', encoding='utf-8') as train_data:
        while True:
            line = train_data.readline()
            if not line:break
            line = line.replace('\n', '')
            data = line.split(',')
            x.append(data[0])
            y.append([int(i) for i in data[1]])

    x_train, x_val, y_train, y_val =\
    train_test_split(x, y, test_size=test_size, random_state=2022)
    
    CWCfans = open('data/test/CWC_answer.txt', 'w', encoding='utf-8')
    CWCfinp = open('data/test/CWC_input.txt', 'w', encoding='utf-8')

    for b_input_ids, b_labels in zip(x_val, y_val):
        CWCfans.write(''.join([str(i) for i in b_labels])+'\n')
        CWCfinp.write(''.join(b_input_ids)+'\n')
    
    print(x_train[:2], x_val[:2], y_train[:2], y_val[:2], len(x_train), len(x_val))
    return x_train, x_val, y_train, y_val

def text_preprocessing(text):
    """
    - Remove entity mentions (eg. '@united')
    - Correct errors (eg. '&amp;' to '&')
    @param    text (str): a string to be processed.
    @return   text (Str): the processed string.
    """
    # Remove '@name'
    text = re.sub(r'(@.*?)[\s]', ' ', text)

    # Replace '&amp;' with '&'
    text = re.sub(r'&amp;', '&', text)

    # Remove trailing whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text

errType = {'R':'ADT', 'S':'MST', 'W':'WORD'}

def write_real_corpus(path, name, train_data, upLimit=5e4):
    n_num = 0
    preProcess(path+'\\'+name, upLimit)
    with open(path+'\\'+name, 'r', encoding='utf-8') as news:
        while True:
            labels = ''
            news_line = news.readline()
            if not news_line:break
            if n_num >= upLimit:break
            for _ in range(MAX_LEN):labels += '0'
            train_data.write(news_line.replace('\n','') + ',' + labels + '\n')#寫新聞句            
            n_num += 1  
    print(n_num)      
        

def write_CWC_corpus(path, trainFile, eType, realFile='newsCorpus.txt', upLimit=5e4, writeReal=False):
    #加入CWC語料
    #用多餘語料，然後產生標記檔 
    err = open(path+'\\對比資料庫.txt', 'r', encoding='utf-8')      
    with open(path+'\\'+trainFile, 'w', encoding='utf-8') as train_data:
            t = {}
            e_num = 0
            max_len = 140
            while True: 
                err_line = err.readline()
                if not err_line:break
                lineTo = err_line.split(',')
                if len(lineTo[1].replace(' ',''))<=3 and len(lineTo[0].replace(' ',''))<=3:continue
                if ('＆' in lineTo[1]) or ('&' in lineTo[1]) or ('+' in lineTo[1])  or ('＄' in lineTo[1]) or ('$' in lineTo[1]):continue
                if len(lineTo[0].replace(' ',''))>140:continue
                if len(lineTo)>0 :        
                    if len(lineTo[9])==3 and lineTo[9]==errType[eType]:
                        if lineTo[9] in t.keys():t[lineTo[9]] += 1
                        else:
                            if eType == 'S':
                                err_sen = lineTo[1].split()
                                cor_sen = lineTo[0].split()
                                #print(err_sen, cor_sen)                        
                                
                                labels = ''
                                for i in range(len(err_sen)):
                                    if cor_sen[i] != err_sen[i]:labels += '1'*len(err_sen[i])
                                    else:labels += '0'*len(err_sen[i])
                                for i in range(max_len-len(lineTo[1].replace(' ',''))):
                                    labels += '0'
                                
                                #train_data.write(lineTo[0].replace(' ','')+','+lineTo[1].replace(' ','')+','+labels+'\n')#寫修正句&偏誤句
                                train_data.write(lineTo[1].replace(' ','')+','+labels+','+lineTo[0].replace(' ','')+'\n')#寫誤代句
                                e_num += 1
                            else:
                                err_sen = lineTo[1].replace(' ','')
                                if err_sen.find(lineTo[11].replace(' ','')) < 0 or err_sen.count(lineTo[11].replace(' ',''))>1:continue
                                if len(lineTo[11])>1:
                                    start, end = err_sen.find(lineTo[11][0]), err_sen.find(lineTo[11][0])+len(lineTo[11])-1
                                    #print(start, end)
                                else:
                                    start, end = err_sen.find(lineTo[11]), err_sen.find(lineTo[11])
                                e_num += 1
                                
                                labels = ''
                                for i in range(len(lineTo[1].replace(' ',''))):
                                    if i >= start and i <= end:labels += '1'
                                    else:labels += '0'
                                for i in range(MAX_LEN-len(lineTo[1].replace(' ',''))):
                                    labels += '0'
                                train_data.write(lineTo[1].replace(' ','')+','+labels+'\n')#寫偏誤句            
            
            err.close()
            if writeReal:write_real_corpus(path, realFile, train_data, upLimit)
            print(t, e_num)

def randomWord(path, realFile, trainFile, upLimit):
    e_num, n_num = 0, 0
    confusionDict, freqDict = {}, {}
    with open(path+"\\字音混淆集.txt",'r',encoding='utf-8') as confusion: #經過字形與字音相似度計算後，為相似字的表
        while True:
            line = confusion.readline().strip()
            if not line:break
            word, confusWord = line.split('　')
            confusionDict[word] = confusWord
    
    with open(path+"\\wordtest4.txt",'r',encoding='utf-8') as table:
        while True:
            line = table.readline().strip()
            if not line:break
            word, freq = line.split(',') 
            freqDict[word] = freq



def write_Wrong_word(path, realFile, trainFile, upLimit):
    e_num = 0
    n_num = 0
    confusion = open(path+"\\字音混淆集.txt",'r',encoding='utf-8') #經過字形與字音相似度計算後，為相似字的表

    dict={}
    while(True):
        line = confusion.readline().strip()
        
        if not line:break
        line = line.split('　')
        if len(line)!=1:dict[line[0]] = line[1]
          
    confusion.close()

    table = open(path+"\\wordtest4.txt",'r',encoding='utf-8') #要挑的字表
    s=0
    dict2={}

    while(True):
        line = table.readline().strip()
        
        if not line:break
        line = line.split(',')
        dict2[s] = line[0]
        s+=1
            
    table.close()

    import random
    def test_(c, prob): #產生錯字
        prob /= 0.15
        if prob < 0.8:
        #if np.random.choice([True, False], p=[0.8, 0.2]): #有0.8的機率是相似的錯字
            line = dict[c].split(' ')
            return line[random.randint(0,len(line)-1)]
        else:    #有0.2的機率是隨機抽字
            a = random.randint(0,len(dict2)-1)
            while(c==dict2[a]):
                a = random.randint(0,len(dict2)-1)
            return dict2[a]
    preProcess(path+'\\'+realFile, upLimit+5e3)
    file = open(path+"\\"+realFile,'r',encoding='utf-8') #要被變成訓練資料的句子，也是校正層解答
    answer_list=''
    a=0
    file2 = open(path+"\\"+trainFile,'w',encoding='utf-8') #產生的訓練資料句

    while True:
        
        line2=''
        answer=[]
        line = file.readline()
        if not line:break
        if e_num >= upLimit: break
        for ch in ['， ', ', ', ' ,', ',']:#
            line = line.replace(ch, '，')
        line = line.replace('︵', '(')
        line = line.replace('︶', ')')
        line = line.replace(':', '：')
        if len(line)>MAX_LEN:continue
        e_num += 1
        for i in line.replace('\n', ''):
            if a==0:
                prob = random.random()
                if prob<=0.15:#有1/15的機率 把這個字當成錯字
                #if np.random.choice([True, False], p=[0.15, 0.85]): 
                    a+=1
                    if dict.get(i) != None: #若這個字不再字表中，則選擇下一個字為錯字 會有a去計數
                        line2+=test_(i, prob)
                        a-=1
                        answer.append(1)
                    else:
                        line2+=i
                        answer.append(0)
                else:
                    line2+=i
                    answer.append(0)
                    
            else:   #若a>0以上，則要一直挑錯字，直到a==0
                if dict.get(i) != None: #
                    line2+=test_(i, prob)
                    a-=1
                    answer.append(1)
                else:
                    line2+=i
                    answer.append(0)
                    
        for j in answer:  #寫入答案
            answer_list+=str(j)
        for i in range(MAX_LEN-len(answer_list)):answer_list += '0'
        file2.write(line2 + ',' + answer_list + '\n')
        answer_list=''
    file2.close()  
    file.close()
    print(e_num, n_num)

if __name__ == "__main__":
    #write_CWC_corpus(path, trainFile, eType, realFile='newsCorpus.txt', upLimit=5e4, writeReal=False)
    #write_Wrong_word(path, realFile, trainFile, upLimit)
    #write_CWC_corpus('data\\train', 'train_dataset.txt', 'R', 'newsCorpus.txt', 8000, True)
    #write_Wrong_word('data\\train', 'newsCorpus.txt', 'train_dataset.txt', 500000)
    pass