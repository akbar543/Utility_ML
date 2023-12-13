from flask import Flask, render_template, request
import os
import text_summary
import imageCompress

import cv2
from keras.models import load_model
import numpy as np
from keras.applications import ResNet50
from keras.optimizers import Adam
from keras.layers import Dense, Flatten,Input, Convolution2D, Dropout, LSTM, TimeDistributed, Embedding, Bidirectional, Activation, RepeatVector,Concatenate
from keras.models import Sequential, Model
# from keras.utils import np_utils
from keras.preprocessing import image, sequence
import cv2
from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm

resnet = ResNet50(include_top=False, weights='imagenet', input_shape=(224,224,3), pooling='avg')
vocab = np.load('vocab.npy', allow_pickle=True)

vocab = vocab.item()

inv_vocab = {v:k for k,v in vocab.items()}

embedding_size = 128
max_len = 40
vocab_size = len(vocab)+1

image_model = Sequential()

image_model.add(Dense(embedding_size, input_shape=(2048,), activation='relu'))
image_model.add(RepeatVector(max_len))
language_model = Sequential()

language_model.add(Embedding(input_dim=vocab_size, output_dim=embedding_size, input_length=max_len))
language_model.add(LSTM(256, return_sequences=True))
language_model.add(TimeDistributed(Dense(embedding_size)))

conca = Concatenate()([image_model.output, language_model.output])
x = LSTM(128, return_sequences=True)(conca)
x = LSTM(512, return_sequences=False)(x)
x = Dense(vocab_size)(x)
out = Activation('softmax')(x)
model = Model(inputs=[image_model.input, language_model.input], outputs = out)

model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])
# model.summary()
model.load_weights('mine_model_weights.h5')

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1

@app.route("/")
def index():
    return render_template('Index.html')

@app.route("/imagecaptioning")
def captioning():
    return render_template("Img_Caption.html")

@app.route('/after', methods=['GET', 'POST'])
def after():
    global model, vocab, inv_vocab, ResNet50
    file = request.files['file1']

    file.save('static/file.jpg')

    img = cv2.imread('static/file.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, (224,224))

    img = np.reshape(img, (1,224,224,3))

    features = resnet.predict(img).reshape(1,2048)

    text_in = ['startofseq']
    final = ''

    print("="*50)
    print("Getting Captions")

    count = 0
    while tqdm(count < 20):

        count += 1

        encoded = []
        for i in text_in:
            encoded.append(vocab[i])

        padded = pad_sequences([encoded], maxlen=max_len, padding='post', truncating='post').reshape(1,max_len)

        sampled_index = np.argmax(model.predict([features, padded]))

        sampled_word = inv_vocab[sampled_index]

        if sampled_word != 'endofseq':
            final = final + ' ' + sampled_word
        if sampled_word == 'endofseq':
            break

        text_in.append(sampled_word)

    return render_template('predict.html', final=final)

@app.route("/imagecompression")
def compression():
    return render_template("Image Compression.html")

@app.route("/summarizer")
def summarizer():
    return render_template("Paraphraser.html")

@app.route("/analyze", methods=["GET", "POST"])
def analyze():
    if request.method== "POST":
        rawtext = request.form['rawtext']
        percent = request.form['para']
        print(percent)
        summary, original_text, len_original, len_summary = text_summary.summarizer(rawtext, percent)
    return render_template("Summary.html", summary=summary, original_text= original_text, len_original=len_original, len_summary=len_summary)

@app.route("/find", methods=['GET', 'POST'])
def find():
    if request.method== "POST":
        image = request.files['my_image']
        img_path = "static/" + image.filename
        root, extension = os.path.splitext(img_path)	
        image.save(img_path)
        print(extension)
        new = "static/output"+extension
        o,n= imageCompress.compress(img_path)
    return render_template("Compress.html", original=img_path, new=new, o=o, n=n)
if __name__ == "__main__":
    app.run(debug=True)

