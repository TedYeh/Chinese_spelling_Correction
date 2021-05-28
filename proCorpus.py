from os import listdir, path, mkdir
import zipfile
import time, json

def preProcess(name='data/train/newsCorpus.txt', num=5e4, path='data/CorpusTXT'):
    n = 0
    max_len = 0
    Wfreq = {}
    txtDir = path
    txtFiles = listdir(txtDir)[:]
    wordfreq = [0, 0, 0]
    with open(name,'w',encoding='utf-8') as f2:
        for fileName in txtFiles:
             with open(txtDir+'/'+fileName, 'r') as f1:
                    while True:
                      line = f1.readline() #讀入第一句
                      if not line:break
                      if n >= num:break
                      while True:
                            secLine = f1.readline()#讀入完整段落
                            line += secLine.replace('\n', '')
                            if ('。' in secLine) or (not secLine):break
                      line = line.replace('\n', '')
                      line = line.replace(line[line.find('。')+1:], '')
                      #print(line)
                      for ch in ['; ',' ? ', '、' ,'、', '、 ']:#
                         line = line.replace(ch, '\n')
                      for ch in ['， ', ', ', ' ,', ',']:#
                         line = line.replace(ch, '，')
                      for ch in [':','︰']:#
                         line = line.replace(ch, '：')
                      for ch in ['   ', '  ', '_2', '_', '「', '」', '– ', '『 ', '』', '.']:
                           line = line.replace(ch, '')
                      line = line.replace('0', '零')
                      line = line.replace('  ', '')
                      #if line[0]==' ':line = line.replace(' ', '')
                      liness = line.split('\n')
                      for s in liness:
                        if '︶' in s or '記者' in s or '中央社' in s:continue
                        if not s:continue
                        
                        for w in s.replace(' ',''):
                            #if (w == '\n') or (w=='「') or (w=='」') or (w=='。'):continue
                            if w in Wfreq.keys():Wfreq[w] += 1
                            else:Wfreq[w] = 1
                        if s[0]==' ':s = s.replace(' ', '')
                        if (len(s.split(' ')) <= 15) or (len(s.split(' ')) > 140):continue
                        if len(s.split(' ')) > max_len:max_len = len(s.split(' '))
                        f2.write(''.join(list(s.replace(' ','')+'\n')))
                        n += 1
                        
                    with open('data/wordFreq.json', 'w') as wJ:
                                json.dump(Wfreq, wJ)
    print(n, wordfreq, max_len)

def proSGML(fileName):    
    sentences, errors, n, lens = [], [], 0, []
    with open(fileName, 'r', encoding='utf-8') as f:
        while True:
            line = f.readline()
            if not line:break
            #if n >= 5:break 
            if '<SENTENCE>' in line:
                error = {}
                while not '</SENTENCE>' in line:

                    line = f.readline()
                    if '<TEXT>' in line:
                        sentences.append(line.replace('<TEXT>', '').replace('</TEXT>', '').replace('\n', '').strip())
                        #lens.append(len(line.replace('<TEXT>', '').replace('</TEXT>', '').replace('\n', '').strip()))
                    if '<MISTAKE>' in line:
                        
                        while not '</MISTAKE>' in line:
                            line = f.readline()
                            if '<LOCATION>' in line:pos = eval(line.replace('<LOCATION>', '').replace('</LOCATION>', '').replace('\n', ''))
                            if '<CORRECTION>' in line:cor = line.replace('<CORRECTION>', '').replace('</CORRECTION>', '').replace('\n', '').strip()
                        error[pos] = cor
                errors.append(error)
                n += 1
        print(n)
    from opencc import OpenCC
    cc = OpenCC('s2twp')
    with open('sgmltrain.txt', 'w', encoding='utf-8') as train_file:
        
        for s, e in zip(sentences, errors):
            label = ['0' for _ in range(140)]
            cor = list(s)
            for loc, c in e.items():
                label[loc-1] = '1'
                cor[loc-1] = c
            train_file.write(cc.convert(s)+','+''.join(label)+','+cc.convert(''.join(cor))+'\n')    

'''
<MISTAKE>
    <LOCATION>14</LOCATION>
    <WRONG>蒽</WRONG>
    <CORRECTION>基</CORRECTION>
</MISTAKE>
''' 

if __name__ == '__main__':
    #preProcess('data/train/testCorpus.txt', 5e4, 'data/CorpusTXT')
    proSGML('train.sgml')    
