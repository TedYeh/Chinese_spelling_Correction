# Chinese-Spell-Detection-For-BERT
目前只有偵測部份，僅偵測該輸入句中是否包含文法錯誤
該程式分訓練及測試模式

## 訓練
` python main.py --mode train --epochs 需要的訓練代數 --module_type 模型類別 --real_num 要產生幾句含別字的錯誤`

ex. 使用 ELECTRA模型訓練40代，並使用50萬別字句訓練

`python main.py --epochs 40 --module_type ELECTRA --real_num 500000 --mode train`


## 測試
` python main.py --mode test --epochs 要讀取訓練第幾代的模型 --module_type 模型類別 --real_num 之前是用多少的別字句訓練的 --test_label 測試標記檔 --test_text 測試文字檔`

ex. 使用訓練了40代，並使用50萬別字句訓練的 ELECTRA 模型測試

`python main.py --mode test --epochs 40 --module_type ELECTRA --real_num 500000 --test_label test_answer_14.txt --test_text test_input_14.txt`

## 其他參數
```
args:
--mode       選擇要訓練還是測試
--train_path 訓練資料檔案目錄
--train_file 訓練資料名稱
--need_real  需不需要正確句
--real_file  正確句名稱
--error_type 錯誤類型(預設為別字)
--real_num   需要多少句的訓練資料
--ckpt_path  checkpoint位置
--epochs     訓練代數
--batch_size 訓練批次量
```

## 檔案
```
BERT_MODEL_Correction (2).ipynb      使用BERT進行校正

BERT_MODEL_Seq2SeqCorrection.ipynb   使用BERT+Pytorch transformer Decoder進行別字校正(未完成)

BERT_MODEL整理後.ipynb                使用BERT作別字偵錯(可用此檔案進行程式測試，比較)
 
dataProcess.py                        產生訓練資料

main.py                               主程式檔

model.py                              BERT模型

proCorpus.py                          資料前處理

test.py                               測試

train.py                              顯示學習曲線
```
