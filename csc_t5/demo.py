import sys
from flask import Flask, request, flash, redirect, url_for, render_template
from loguru import logger
import asyncio
sys.path.append("..")
from t5.t5_model import T5Model
import os
os.environ['JOBLIB_TEMP_FOLDER'] = '/tmp'

app = Flask(__name__)

output_dir = './outputs/prompt_1m'
csc = T5Model('t5', output_dir, args={
    "eval_batch_size": 1}, cuda_device=-1, evaluate=True)

#output_dir = './outputs/prompt_cgedit'
#csc_large = T5Model('t5', output_dir, args={"eval_batch_size": 1}, cuda_device=-1, evaluate=True)

output_dir = './outputs/prompt_cgedit'
cged = T5Model('t5', output_dir, args={"eval_batch_size": 1}, cuda_device=-1, evaluate=True)

help = """
You can request the service by HTTP get: <br> 
   http://0.0.0.0:5001/macbert_correct?text=我从北京南做高铁到南京南<br>
   
or HTTP post with json: <br>  
   {"text":"xxxx"} <p>
Post example: <br>
  curl -H "Content-Type: application/json" -X POST -d '{"text":"我从北京南做高铁到南京南"}' http://0.0.0.0:5001/macbert_correct
"""

def compare_str(src, tar): #for csc - compare the input sentence and prediction then tag the different part(location) 
    loc = 0
    loc_list = []
    for s, t in zip(src, tar):
        if s != t: loc_list.append(loc)
        loc += 1
    return loc_list

async def do_correct(model, text):
    return model.predict(text)

@app.route('/t5_correct', methods=['POST', 'GET'])
async def t5_correct():
    global csc, cged
    diff_loc_list = []
    model = {'Spelling-T5-Base': csc, 'Grammar-T5-Base': cged}    
    if request.method == 'POST':
        text = request.form['input'] #取得輸入框內容
        select_model = request.form['models'] #取得使用哪個模型
        prompt = {'Spelling-T5-Base': '糾正句子中的錯字：', 'Grammar-T5-Base': request.form['prompt']}
        
        #在後端顯示輸入句、選擇模型及使用的提示
        logger.info("Received data: {}".format(text))
        logger.info("Use model: {}".format(select_model))
        logger.info("Prompt: {}".format(prompt[select_model]))
        
        #將提示和輸入句輸入至模型進行預測
        results = await do_correct(model[select_model], [prompt[select_model] + text + "_輸出句："])
        if len(text) == len(results[0]): diff_loc_list = compare_str(text, results[0])

        #將模型輸出放至html檔
        return render_template('home.html', inp_text=text, results=results[0], diff_loc_list=diff_loc_list)
    src = '為了降低少子化，政府可以堆動獎勵生育的政策。'
    tar = '為了降低少子化，政府可以推動獎勵生育的政策。'
    diff_loc_list = compare_str(src, tar)
    return render_template('home.html', inp_text=src, results=tar, diff_loc_list=diff_loc_list)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8787, debug=True)
