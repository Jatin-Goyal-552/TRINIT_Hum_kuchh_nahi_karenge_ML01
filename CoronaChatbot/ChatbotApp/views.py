from django.shortcuts import render

# Create your views here.
from django.urls import reverse
from .models import *
# Create your views here.
from django.shortcuts import render, HttpResponse
import pickle
import json
import random
import numpy as np
import nltk
import pandas as pd
# nltk.download('punkt')
# nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
import keras
import random


# disease_model= pickle.load(open('C://Users//LENOVO//projects//Health Care Chatbot//notebook//Multinomial_classifier_disease.pkl','rb'))
# disease_tokenizer = pickle.load(open('C://Users//LENOVO//projects//Health Care Chatbot//notebook//tf_idf_vectorizer_disease.pkl','rb'))
model = keras.models.load_model('C://Users//LENOVO//projects//TRI Nit Hackathon//chatbot//chatbot_model.h5')
intents = json.loads(open('C://Users//LENOVO//projects//TRI Nit Hackathon//chatbot//intents.json').read())
words = pickle.load(open('C://Users//LENOVO//projects//TRI Nit Hackathon//chatbot//words.pkl','rb'))
classes = pickle.load(open('C://Users//LENOVO//projects//TRI Nit Hackathon//chatbot//classes.pkl','rb'))
print("---------------------You are set to go.--------------------------")
# flag=False
# prec,desc,sym=False,False,False
# temp_disease=''
# all_symptoms=''

def chatbot(request):
    return render(request,'chatbot.html')

def clean(text):
    text = text.lower() 
    text = text.split()
    text = ' '.join(text)
    return text

def clean_up_sentence(sentence):
    lemmatizer = WordNetLemmatizer()
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    p = bow(sentence, words,show_details=True)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result,tag

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res,tag = getResponse(ints, intents)
    return res,tag

def home(request):
    return render(request,'home.html')

def chatbot(request):
    return render(request,'corona_chatbot.html')

def predict_chat(request):
    pred="please type something"
    tag=""
    global prec,desc,sym,flag,temp_disease,all_symptoms
    if request.method == 'POST':
        print('hello')
        chat=request.POST['operation']
        pred,tag=chatbot_response(chat)
        # pred,tag=chatbot_response(chat)
    return HttpResponse(json.dumps({'ans':pred}), content_type="application/json")

def xray(request):
    return render(request,'corona_Xray_form.html')