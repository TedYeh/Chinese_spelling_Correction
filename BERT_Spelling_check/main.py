from argparse import ArgumentParser
import torch
from transformers import BertForMaskedLM
from dataProcess import get_data, write_CWC_corpus, write_real_corpus, write_Wrong_word
from model import preprocessing_for_bert, get_dataloader, initialize_model, train
from train import get_train_state
from test import show_confusion, get_test_result, get_test_data
import random
import numpy as np

'''
args:
--mode train test
--train_path
--train_file
--need_real
--real_file
--error_type
--real_num
--ckpt_path
--epochs
--batch_size
'''
#write_CWC_corpus(path, trainFile, eType, realFile='newsCorpus.txt', upLimit=5e4, writeReal=False)
#write_Wrong_word(path, realFile, trainFile, upLimit)
#preprocessing_for_bert(data)
#get_dataloader(train_inputs, y_train, val_inputs, y_val, batch_size)
#get_train_state(losses)
#initialize_model(train_dataloader = [], epochs=4)
#train(model, optimizer, scheduler, train_dataloader, ckptDir, eType, epochs=4)

def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False

parser = ArgumentParser()
parser.add_argument('--mode', dest = 'mode', default = 'train') #選擇訓練或測試模式{train, test}
parser.add_argument('--train_path', dest = 't_path', default = 'data\\train') #訓練語料位置
parser.add_argument('--train_file', dest = 't_file', default = 'train_dataset.txt') #訓練語料檔案名稱
parser.add_argument('--need_real', dest = 'need_real', type=str2bool, nargs='?', const=True, default = True) #是否需要真實語料
parser.add_argument('--real_file', dest = 'r_file', default = 'newsCorpus.txt') #真實資料位置
parser.add_argument('--error_type', dest = 'type', default = 'WORD') #句法錯誤類型，有三種:{'R':多餘, 'S':誤代, 'WORD':錯字}
parser.add_argument('--real_num', dest = 'r_num', default = 2e5, type = int)
parser.add_argument('--ckpt_path', dest = 'ckpt_path', default = 'ckpt') #checkpoint路徑
parser.add_argument('--test_path', dest = 'test_path', default = 'data\\test') #測試檔案路徑
parser.add_argument('--test_label', dest = 'label', default = 'test_answer_14.txt')#測試檔案標記檔
parser.add_argument('--test_text', dest = 'text', default = 'test_input_14.txt')#測試檔案文字檔
parser.add_argument('--epochs', dest = 'epochs', default = 10, type = int) #訓練迭代次數
parser.add_argument('--batch_size', dest = 'b_sz', default = 64, type = int) #batch size
parser.add_argument('--old_epochs', dest = 'oldEpochs', default = 10, type = int)
parser.add_argument('--module_type', dest = 'md_type', default = 'normal')
args = parser.parse_args()

#使用範例：
# 用50萬句語料訓練ELECTRA 模型 40代：python main.py --epochs 40 --module_type ELECTRA --real_num 500000 --mode train
# 用第40代的ELECTRA進行測試：        python main.py --epochs 40 --module_type ELECTRA --real_num 500000 --mode test --test_label test_answer_14.txt --test_text test_input_14.txt
 
print(args)
if args.mode == 'train':
    same_seeds(0)
    if args.type != 'WORD':#以CWC語料或自行產生之錯字語料訓練
        write_CWC_corpus(args.t_path, args.t_file, args.type, args.r_file, args.r_num, args.need_real) 
    else:
        write_Wrong_word(args.t_path, args.r_file, args.t_file, args.r_num)

    #取得訓練及驗證資料
    x_train, x_val, y_train, y_val = get_data(args.t_path, args.t_file) 

    #前處理，將文字處理為Bert Embedding
    train_inputs = preprocessing_for_bert(x_train, module_type=args.md_type) 
    val_inputs = preprocessing_for_bert(x_val, module_type=args.md_type)

    #取得dataloader
    train_dataloader, val_dataloader = get_dataloader(train_inputs, y_train, val_inputs, y_val, args.b_sz)

    #模型初始化，需選擇要以哪種模型訓練
    bert_classifier, optimizer, scheduler = initialize_model(args.md_type, train_dataloader, args.epochs)

    #開始訓練
    losses = train(bert_classifier, optimizer, scheduler, train_dataloader, args.ckpt_path, args.type, args.r_num, args.epochs, module_type=args.md_type)
    
    #取得學習曲線
    get_train_state(args.t_path, losses, args.type, args.epochs)

    #印出驗證結果及confusion matrix
    y_real_s, y_pred_s, y_real_c, y_pred_c = get_test_result(bert_classifier, val_dataloader, args.r_num, args.type, args.epochs, args.md_type)
    show_confusion(y_real_s, y_pred_s, y_real_c, y_pred_c)

elif args.mode == 'test':
    '''讀取模型並載入checkpoint'''
    bert_classifier, _, _ = initialize_model(module_type=args.md_type) 
    checkpoint = torch.load(args.ckpt_path+'/M_{}_{}_E_{}_N_{}.h5'.format(args.md_type, args.type, args.epochs, args.r_num))
    #checkpoint = torch.load(args.ckpt_path+'/WORD_E_10.h5'.format(args.type, args.epochs, args.r_num))
    bert_classifier.load_state_dict(checkpoint)
    bert_classifier.eval()

    '''載入測試檔案'''
    test_dataloader = get_test_data(args.test_path, args.label, args.text, args.b_sz)

    '''測試並顯示結果'''
    y_real_s, y_pred_s, y_real_c, y_pred_c = get_test_result(bert_classifier, test_dataloader, args.r_num, args.type, args.epochs, args.md_type)
    show_confusion(y_real_s, y_pred_s, y_real_c, y_pred_c)
elif args.mode == 'test_all':
    bert_classifier, _, _ = initialize_model(module_type=args.md_type)
    checkpoint = torch.load(args.ckpt_path+'/M_ELECTRA_WORD_E_20_N_500010.h5')
    bert_classifier.load_state_dict(checkpoint)
    PRETRAINED_MODEL_NAME = "hfl/chinese-roberta-wwm-ext"
    bert_correcter = BertForMaskedLM.from_pretrained(PRETRAINED_MODEL_NAME)
    test_dataloader = get_test_data(args.test_path, args.label, args.text, args.b_sz)
    y_real_s, y_pred_s, y_real_c, y_pred_c = get_test_result(bert_classifier, test_dataloader, args.r_num, args.type, args.epochs, args.md_type, bert_correcter)
    show_confusion(y_real_s, y_pred_s, y_real_c, y_pred_c)

'''
elif args.mode == 'old':
    input()
    x_train, x_val, y_train, y_val = get_data(args.t_path, args.t_file)
    train_inputs = preprocessing_for_bert(x_train)
    val_inputs = preprocessing_for_bert(x_val)
    train_dataloader, val_dataloader = get_dataloader(train_inputs, y_train, val_inputs, y_val, args.b_sz)
    bert_classifier, optimizer, scheduler = initialize_model(train_dataloader, args.epochs)
    checkpoint = torch.load(args.ckpt_path+'/{}_E_{}_N_{}.h5'.format(args.type, args.oldEpochs, args.r_num))
    bert_classifier.load_state_dict(checkpoint)
    bert_classifier.train()
    losses = train(bert_classifier, optimizer, scheduler, train_dataloader, args.ckpt_path, args.type, args.r_num, args.oldEpochs, True, args.epochs)
'''   


