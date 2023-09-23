import sys
from flask import Flask, request, flash, redirect, url_for, render_template
from loguru import logger
import asyncio
sys.path.append("..")
from t5.t5_model import T5Model

app = Flask(__name__)

help = """
You can request the service by HTTP get: <br> 
   http://0.0.0.0:5001/macbert_correct?text=我从北京南做高铁到南京南<br>
   
or HTTP post with json: <br>  
   {"text":"xxxx"} <p>
Post example: <br>
  curl -H "Content-Type: application/json" -X POST -d '{"text":"我从北京南做高铁到南京南"}' http://0.0.0.0:5001/macbert_correct
"""

async def do_correct(text):
    return model.predict(text)

@app.route('/t5_correct', methods=['POST', 'GET'])
async def t5_correct():
    global model
    if request.method == 'POST':
        text = request.form['input']
        logger.info("Received data: {}".format(text))
        results = await do_correct(['糾正句子中的錯字：' + text + "_輸出句："])
        return render_template('home.html', inp_text=text, results=results)
    return render_template('home.html', inp_text='', results='')


if __name__ == '__main__':
    output_dir = './outputs/prompt_csc_3.5e5_prompt_v1_newcorpus'
    model = T5Model('t5', output_dir, args={"eval_batch_size": 16})
    app.run(host="0.0.0.0", port=8787, debug=True, use_reloader=True)