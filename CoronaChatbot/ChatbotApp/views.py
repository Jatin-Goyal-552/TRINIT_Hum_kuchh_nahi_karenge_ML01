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
from .forms import corona_xray_form
from textblob import TextBlob
import keras
import cv2
from collections import Counter
import os
from time import sleep
# disease_model= pickle.load(open('C://Users//LENOVO//projects//Health Care Chatbot//notebook//Multinomial_classifier_disease.pkl','rb'))
# disease_tokenizer = pickle.load(open('C://Users//LENOVO//projects//Health Care Chatbot//notebook//tf_idf_vectorizer_disease.pkl','rb'))
model = keras.models.load_model('C://Users//LENOVO//projects//TRI Nit Hackathon//chatbot//chatbot_model3.h5')
intents = json.loads(open('C://Users//LENOVO//projects//TRI Nit Hackathon//chatbot//intents.json').read())
words = pickle.load(open('C://Users//LENOVO//projects//TRI Nit Hackathon//chatbot//words.pkl','rb'))
classes = pickle.load(open('C://Users//LENOVO//projects//TRI Nit Hackathon//chatbot//classes.pkl','rb'))
model0 = keras.models.load_model("C://Users//LENOVO//projects//tri_nit_xray_models//corona_model0.h5")
model1 = keras.models.load_model("C://Users//LENOVO//projects//tri_nit_xray_models//corona_model1.h5")
model2 = keras.models.load_model("C://Users//LENOVO//projects//tri_nit_xray_models//corona_model2.h5")
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
    ignore_words=['covid','corona','covid-19','19','breastfeed','newborn','unborn','viruses','viruse','varient']

    lemmatizer = WordNetLemmatizer()
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words2=[]
    for a in sentence_words:
        if a.lower() not in ignore_words:
            sentence_words2.append(str(TextBlob(a).correct()))
        else:
            sentence_words2.append(a)
    sentence_words=sentence_words2 
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

def preprocess(imagePath):
    data1 = []
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image,(224,224))
    data1.append(image)
    data1 = np.array(data1)/255.0
    
    return data1

def predict(m1,m2,m3,imagePath):
    im =preprocess(imagePath)
    models = []
    predictions = []
    model_list = [m1,m2,m3]
    for i in range(3):
        model =model_list[i]
        models.append(model)
        predictions.append(model.predict(im))
        
    my_list = []
    for dd in predictions:
        if dd>.5:
            my_list.append(1)
        else:
            my_list.append(0)
    print("all predictions",my_list)
        
    cn = Counter(my_list)
    value,count = cn.most_common()[0]
    if value == 1:
        return "This X-ray is Covid Postitive"
    else:
        return "This X-ray is Covid Negative"
        

def xray(request):
    form=corona_xray_form()
    
    if request.method == 'POST':
        form=corona_xray_form(request.POST,request.FILES)
        if form.is_valid():
            
            form.save()
        else:
            print(form.errors)
        # url=str(form.image)
        url=str(form.cleaned_data["image"])
        print("url",url)
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # url="http://localhost:8000/media/"+url
        sleep(5)
        # model=keras.models.load_model("corona_model.h5")
        image_location=os.path.join("media",url)
        # img = load_img(image_location, grayscale=False, target_size=(150, 150,3))
        # # response = requests.get(url)
        # # img = Image.open(BytesIO(response.content))
        # img = img_to_array(img)
        # img= img.reshape(1, 150, 150, 3)
        # img = img.astype('float32')
        # img = img / 255.0
        # print(img)
        
        # pred=model.predict(img)[0][0]
        # print("pred",pred)
        # if pred>0.5:
        #     prediction="This image do not have corona virus."
        # else:
        #     prediction="This image have corona virus."
        # print("image_pic_url",brain_mri)
        prediction=predict(model0,model1,model2,image_location)
        corona = corona_xray.objects.all()
        # mri=brain.filter(brain_mri_id=5)
        # print("mri",corona)
        # print("mri",mri)
        sorted_xray= corona_xray.objects.order_by('corona_id').reverse()
        print("*****************************") 
        print("sorted",sorted_xray[0].corona_id)
        # print("image_pic_url",brain_mri)
        # image_location_path=os.path.join(BASE_DIR,"media",url)
        # print("image_location_",image_location_path)
        print("nnnnn")
        last=sorted_xray[0].corona_id
        xray=corona.filter(corona_id=last)
        print("mri",xray)
        return render(request, 'corona_xray_result.html',{"prediction": prediction,"xray":xray})
    return render(request,'corona_Xray_form.html',{"form": form})