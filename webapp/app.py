import flask
from flask import request, redirect, url_for
from flask import send_file

import pickle
import numpy as np
import librosa
import tensorflow as tf

# load_options=tf.saved_model.LoadOptions(
#     experimental_io_device='/job:localhost'
# )
model_text=pickle.load(open('models/lr.pkl','rb'))
vector=pickle.load(open('models/vector.pkl','rb'))
normalizer=pickle.load(open('models/normalizer.pkl','rb'))
selector=pickle.load(open('models/selector.pkl','rb'))
model_audio=tf.keras.models.load_model('models/new_model')

def get_text_emotion(text):
  data=np.array([text])
  data=vector.transform(data)
  data=data.toarray()
  data=normalizer.transform(data) 
  data=selector.transform(data)
  emotion=model_text.predict(data)
  return emotion[0]

def get_audio_emotion(path):
  x, sample_rate = librosa.load(path, res_type='kaiser_fast',offset=0.5,duration=3)
  mfcc = np.mean(librosa.feature.mfcc(y=x, sr=sample_rate,n_mfcc=40).T,axis=0)
  data=[]
  data.append(mfcc)
  data=np.expand_dims(data,axis=2)
  label=['non-depressed','depressed']
  emotion=[np.argmax(model_audio.predict(data))]
  return label[emotion[0]]

app = flask.Flask(__name__, template_folder='templates')
@app.route('/')
def main():
    return(flask.render_template('main.html'))
if __name__ == '__main__':
    app.run()

@app.route('/', methods=['POST'])
def upload_file():
    file = request.files['file']
    if file.filename != '':
        file.save(file.filename)
    file_type=file.filename.split('.')[-1]
    if(file_type=='txt'):
        f=open(file.filename,'r')
        txt=f.read()
        emotion=get_text_emotion(txt)
        data=[{"text":txt,"label":emotion}]
        return flask.render_template('resultText.html',data=data)
    else:
        f=open(file.filename,'rb')
        emotion=get_audio_emotion(f)
        # display(Audio(input, autoplay=False))
        print(emotion)
        data=[{"file":file.filename,"label":emotion}]
        return flask.render_template('resultAudio.html',data=data)
@app.route('/', methods=['GET'])
def hello():
    return(flask.render_template('main.html'))
    