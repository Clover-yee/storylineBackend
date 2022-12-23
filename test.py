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
if __name__ == '__main__':
    text = "The evening altogether passed off pleasantly to the whole family. Mrs. Bennet had seen her eldest " \
           "daughter much admired by the Netherfield party. Mr. Bingley had danced with her twice, and she had been " \
           "distinguished by his sisters. Jane was as much gratified by this as her mother could be, though in a " \
           "quieter way. Elizabeth felt Jane's pleasure. Mary had heard herself mentioned to Miss Bingley as the most " \
           "accomplished girl in the neighbourhood; and Catherine and Lydia had been fortunate enough to be never " \
           "without partners, which was all that they had yet learnt to care for at a ball. They returned therefore, " \
           "in good spirits to Longbourn, the village where they lived, and of which they were the principal " \
           "inhabitants. They found Mr. Bennet still up. With a book, he was regardless of time; and on the present " \
           "occasion he had a good deal of curiosity as to the event of an evening which had raised such splendid " \
           "expectations. He had rather hoped that all his wife's views on the stranger would be disappointed; but he " \
           "soon found that he had a very different story to hear. "

    res = summarizer(text)
    print(res, 'before')
