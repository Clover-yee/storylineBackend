from flask import Flask, jsonify, request
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline, BertTokenizerFast, \
    AutoModelForSeq2SeqLM
import json
import numpy as np

# identifier = pipeline(task="token-classification")
tokenizer_distilbart = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
model_distilbart = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-cnn-12-6")
summarizer = pipeline("summarization", model=model_distilbart, tokenizer=tokenizer_distilbart)

# summarizer = pipeline("summarization", model="facebook/bart-large-cnn")


tokenizerChinese = BertTokenizerFast.from_pretrained('bert-base-chinese')
identifierChinese = pipeline(task="token-classification", model="ckiplab/bert-base-chinese-ner",
                             tokenizer=tokenizerChinese)

tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")

identifierFirst = pipeline(task="token-classification", model=model, tokenizer=tokenizer, aggregation_strategy="first")
identifierMax = pipeline(task="token-classification", model=model, tokenizer=tokenizer, aggregation_strategy="max")
identifierNone = pipeline(task="token-classification", model=model, tokenizer=tokenizer, aggregation_strategy="none")
identifierSimple = pipeline(task="token-classification", model=model, tokenizer=tokenizer,
                            aggregation_strategy="simple")
identifierAverage = pipeline(task="token-classification", model=model, tokenizer=tokenizer,
                             aggregation_strategy="average")

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False


@app.route('/classification', methods=['POST'])
def classification():
    text = request.json.get('text')
    strategy = request.json.get('strategy')
    if strategy == 'first':
        res = identifierFirst(text)
    elif strategy == 'max':
        res = identifierMax(text)
    elif strategy == 'none':
        res = identifierNone(text)
    elif strategy == 'simple':
        res = identifierSimple(text)
    elif strategy == 'average':
        res = identifierAverage(text)

    res = np.array(res).tolist()
    print(res)
    return json.dumps(res, cls=MyEncoder)


@app.route('/summarization', methods=['POST'])
def summarization():
    text = request.json.get('text')
    print(len(text))
    resStr = ''
    resList = []
    n_content = text[:]
    list = []
    while True:
        aim_file = n_content[:1024]
        list.append(aim_file)
        n_content = n_content.strip(aim_file)
        if len(n_content) == 0:
            break

    print(len(list))
    for each in list:
        res = summarizer(each, max_length=30, min_length=15, do_sample=False)
        resList.append(res[0])
        resStr += res[0]['summary_text'] + '.'
        print(res)

    # res = summarizer(text, max_length=50, min_length=30, do_sample=False)
    # res = np.array(res).tolist()
    return json.dumps(resList, cls=MyEncoder)


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


CORS(app, resources=r'/*')
if __name__ == '__main__':
    app.run()
