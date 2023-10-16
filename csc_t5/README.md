# CGEDit - Chinese Grammatical Error Diagnosis by Task-Specific Instruction Tuning

## Overview
- We propose a instruction-tuned T5 to do correction of Chinese Grammatical Error:
    - **Spelling-T5-Base** instruction-tuned on over 1M sentences in traditional mandarin.
    - **Grammar-T5-Base** instruction-tuned on 5 tasks and over 150k sentences in traditional mandarin.

## Dataset Overview

- The dataset includes five tasks(150k sentences):
    - Spelling Correction: 36,402 sentences
    - Grammatical Error Correction: 34,868 sentences
    - Text Paraphrasing: 13,936 sentences
    - Text Simplification: 21,061 sentences
    - Multi-Task: 43,733 sentences

## File

```
demo.py                         建立flask網頁系統
training_zh_prompt_model_csc.py 訓練主程式
t5/t5_model.py                  定義T5 model
t5/t5_utils.py                  定義評估指標及資料前處理演算法
t5/config/model_args.py         定義model的各項參數
templates/                      網站html檔
static/                         Boostrap 5, JQuery等函式庫
data/                           訓練、測試資料及字音形混淆集
```