from flask import Flask, request
from flask_restful import Resource, Api
from flask_cors import CORS
import numpy as np
import dill as pickle
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from keras.preprocessing.text import Tokenizer
import speech_recognition as sr
import moviepy.editor as mp
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))
from keras.preprocessing.sequence import pad_sequences
import librosa
import subprocess

app = Flask(__name__)
api = Api(app)
CORS(app)


class facefeature(Resource):
    def post(self):
        try:
            url = request.args.get('Url')
            return {'data' : self.predict(url)}
        except Exception as e:
            return {'data': 'An Error Occurred during fetching Api : '+ str(e)}, 400

    def predict(self, y):
      Filepath = "mysite/Data"
      vd = y
      PADDING = 40
      NumberofFrames = 50
      emotion_labels = self.get_labels()
      arry = {}

      vidcap = cv2.VideoCapture(vd)

      success,image = vidcap.read()
      frame_count = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
      print("Frame Count: ", frame_count)
      count = 0
      cascPath= Filepath+"/abc.xml"
      faceCascade = cv2.CascadeClassifier(cascPath)

      while vidcap.isOpened():
          score = 0
          success,image = vidcap.read()

          if success:
              if frame_count > NumberofFrames+1:
                count += frame_count/(NumberofFrames+1) # i.e. at 30 fps, this advances one second
              else:
                count += 1
              vidcap.set(cv2.CAP_PROP_POS_FRAMES, count)
              gray_image_array = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
              faces = faceCascade.detectMultiScale(
              gray_image_array,
              scaleFactor=1.1,
              minNeighbors=5,
              minSize=(30, 30))

              if len(faces) == 1:
                gray_img = self.pad(gray_image_array)

                emotions = []
                for face_coordinates in faces:
                    face_coordinates = self.tosquare(face_coordinates)
                    x1, x2, y1, y2 = self.apply_offsets(face_coordinates)

                    # adjust for padding
                    x1 += PADDING
                    x2 += PADDING
                    y1 += PADDING
                    y2 += PADDING
                    x1 = np.clip(x1, a_min=0, a_max=None)
                    y1 = np.clip(y1, a_min=0, a_max=None)

                    #gray_face = gray_img[max(0, y1 - PADDING):y2 + PADDING,
                    #                    max(0, x1 - PADDING):x2 + PADDING]
                    #gray_face = gray_img[y1:y2, x1:x2]

                    emotion_model = Filepath+"/model1.hdf5"
                    model = load_model(emotion_model, compile=compile)
                    model.make_predict_function()

                    try:
                      gray_face = cv2.resize(gray_img, model.input_shape[1:3])
                    except Exception as e:
                      print("Cannot resize "+str(e))
                      continue

                    # Local Keras model
                    #gray_face = self.preprocess_input(gray_face, True)
                    gray_face = np.expand_dims(np.expand_dims(gray_face, 0), -1)

                    emotion_prediction = model.predict(gray_face)[0]
                    labelled_emotions = {
                        emotion_labels[idx]: round(float(score), 2)
                        for idx, score in enumerate(emotion_prediction)
                    }

                    emotions.append(
                        dict(box=face_coordinates, emotions=labelled_emotions)
                    )
                top_emotions  = [max(e["emotions"], key=lambda key: e["emotions"][key]) for e in emotions]
                if len(top_emotions):
                  for top_emotion in emotions[0]["emotions"]:
                    if top_emotion in arry.keys():
                      arry.update({top_emotion: arry[top_emotion] + emotions[0]["emotions"][top_emotion]})
                    else:
                      arry[top_emotion] = score

          else:
              vidcap.release()
              break
      if len(arry) == 0:
        return "neutral"
      else:
        return max(arry, key=arry.get)

    def transform(self, X, y):
      return "fit"

    def get_labels(self):
      return {
                0: "angry",
                1: "disgust",
                2: "fear",
                3: "happy",
                4: "sad",
                5: "surprise",
                6: "neutral",
            }

    def tosquare(self, bbox):
            """Convert bounding box to square by elongating shorter side."""
            x, y, w, h = bbox
            if h > w:
                diff = h - w
                x -= diff // 2
                w += diff
            elif w > h:
                diff = w - h
                y -= diff // 2
                h += diff
            if w != h:
                print(f"{w} is not {h}")

            return (x, y, w, h)

    def apply_offsets(self, face_coordinates):
      x, y, width, height = face_coordinates
      x_off, y_off = (10, 10)
      return (x - x_off, x + width + x_off, y - y_off, y + height + y_off)

    def preprocess_input(self, x, v2=False):
            x = x.astype("float32")
            x = x / 255.0
            if v2:
                x = x - 0.5
                x = x * 2.0
            return x

    def pad(self, image):
            PADDING = 40
            row, col = image.shape[:2]
            bottom = image[row - 2 : row, 0:col]
            mean = cv2.mean(bottom)[0]

            padded_image = cv2.copyMakeBorder(
                image,
                top = PADDING,
                bottom = PADDING,
                left = PADDING,
                right= PADDING,
                borderType=cv2.BORDER_CONSTANT,
                value=[mean, mean, mean],
            )
            return padded_image

class tonefeature(Resource):
    def post(self):
        try:
            url = request.args.get('Url')
            return {'data' : self.toneAnalyze(url)}
        except Exception as e:
            return {'data': 'An Error Occurred during fetching Api : '+ str(e)}, 400

    def toneAnalyze(self, clip):
          self.convert_video_to_audio_ffmpeg(clip)
          path = "audiopath.wav"
          data, sample_rate = librosa.load(path)
          feature = self.get_features(path,sample_rate)
          model = tf.keras.models.load_model("mysite/Data/SERf2-100_model.h5")
          pred_test = model.predict(feature)
          with open("mysite/Data/encoder","rb") as f:
            encoder = pickle.load(f)
          y_pred = encoder.inverse_transform(pred_test)
          return y_pred[0][0]


    def convert_video_to_audio_ffmpeg(self, video_file, output_ext="wav"):
        subprocess.call(["ffmpeg", "-y", "-i", video_file, f"{'audiopath'}.{output_ext}"], stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)

    def noise(self, data):
        noise_amp = 0.035*np.random.uniform()*np.amax(data)
        data = data + noise_amp*np.random.normal(size=data.shape[0])
        return data

    def stretch(self, data, rate=0.8):
        return librosa.effects.time_stretch(data, rate)

    def shift(self, data):
        shift_range = int(np.random.uniform(low=-5, high = 5)*1000)
        return np.roll(data, shift_range)

    def pitch(self, data, sampling_rate, pitch_factor=0.7):
        return librosa.effects.pitch_shift(data, sampling_rate, pitch_factor)

    def extract_features(self, data, sample_rate):
        # ZCR
        result = np.array([])
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
        result=np.hstack((result, zcr))
        # Chroma_stft
        stft = np.abs(librosa.stft(data))
        chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
        result = np.hstack((result, chroma_stft))

        # MFCC
        mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
        result = np.hstack((result, mfcc))

        # Root Mean Square Value
        rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
        result = np.hstack((result, rms))

        # MelSpectogram
        mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
        result = np.hstack((result, mel))

        return result

    def get_features(self, path, sample_rate):
        # duration and offset are used to take care of the no audio in start and the ending of each audio files as seen above.
        data, sample_rate = librosa.load(path, duration=2.5, offset=0.6)

        # without augmentation
        res1 = self.extract_features(data, sample_rate)
        result = np.array(res1)

        # data with noise
        noise_data = self.noise(data)
        res2 = self.extract_features(noise_data, sample_rate)
        result = np.vstack((result, res2)) # stacking vertically

        # data with stretching and pitching
        new_data = self.stretch(data)
        data_stretch_pitch = self.pitch(new_data, sample_rate)
        res3 = self.extract_features(data_stretch_pitch, sample_rate)
        result = np.vstack((result, res3)) # stacking vertically

        return result

class textfeature(Resource):
    def post(self):
        try:
            url = request.args.get('Url')
            return {'data' : self.speechpred(url)}
        except Exception as e:
            return {'data': 'An Error Occurred during fetching Api : '+ str(e)},400

    def speechpred(self, clip):
      clip = mp.VideoFileClip(clip)
      text = (self.videotoText(clip))
      STOPWORDS = set(stopwords.words('english'))
      text = ' '.join(word for word in text.split() if word not in STOPWORDS)
      text = [text]

      # The maximum number of words to be used. (most frequent)
      MAX_NB_WORDS = 50000
      # Max number of words in each complaint.
      MAX_SEQUENCE_LENGTH = 250

      tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)

      tokenizer.fit_on_texts(text)
      #word_index = tokenizer.word_index
      text = (self.Convert(text))

      seq = tokenizer.texts_to_sequences(text)
      padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
      loaded_model = tf.keras.models.load_model("mysite/Data/my_model (1).h5")

      #loaded_model = pickle.load(open("/content/my_model.h5", 'rb'))
      pred = loaded_model.predict(padded)
      labels = ['happy', 'fear', 'angry', 'sad', 'neutral','disgust','surprise']
      return str(labels[np.argmax(pred)])

    def videotoText(self, clip):
        clip.audio.write_audiofile(r"mysite/Data/audio.wav")
        r = sr.Recognizer()
        audio = sr.AudioFile("mysite/Data/audio.wav")
        with audio as source:
          audio_file = r.record(source)
        result = r.recognize_google(audio_file)
        return str(result)

    def Convert(self, string):
      list1=[]
      list1[:0]=string
      return list1

class aggregate(Resource):
    def post(self):
        try:
            url = request.args.get('Url')

            face = facefeature()
            faceft1 = facefeature.predict(face, url)
            text = textfeature()
            textft1 = textfeature.speechpred(text, url)
            tone = tonefeature()
            toneft1 = tonefeature.toneAnalyze(tone, url)
            return {'data' : self.aggregate3models(faceft1, textft1, toneft1)}
        except Exception as e:
            return {'data': 'An Error Occurred during fetching Api : '+ str(e)}, 400

    def aggregate3models(self, facefeature, tonefeature, textfeature):
        Filepath = "mysite/Data"
        emotion_keys = {"happy": 0, "sad": 1, "surprise": 2, "fear": 3, "neutral": 4, "disgust": 5, "angry": 6}
        emotion_keys_swapped = dict([(value, key) for key, value in emotion_keys.items()])
        input_array_from_models = [facefeature, tonefeature, textfeature]
        prediction_x = tf.reshape(tf.one_hot(tf.convert_to_tensor([emotion_keys[item] for item in input_array_from_models]), depth=7), [1, 21])
        model = tf.keras.models.load_model(Filepath+'/my_model.h5')
        # make a prediction from our trained model and print probabilities
        probabilities = model.predict(prediction_x)
        new = {}
        for i in range(len(probabilities[0])):
            new[emotion_keys_swapped[i]] = probabilities[0][i]
        maxVal = max(new, key=new.get)
        final = str(maxVal)+' '+str(new[maxVal])
        return final

class activity(Resource):
    def post(self):
        try:
            userId = request.args.get('User')
            emotion = request.args.get('Emotion')
            user_emotions = ['angry', 'neutral', 'fear', 'disgust', 'happy', 'sad', 'surprise']
            if emotion not in user_emotions:
                return {'data' : "Invalid emotion"},400
            else:
                prediction = self.predict_emotion_activity(emotion, user_emotions)

                response = {
                    "userId": userId,
                    "emotion": emotion,
                    "predicted_activity_id": prediction
                }

                return {'data' : response}

        except Exception as e:
            return {'data': 'An Error Occurred during fetching Api : '+ str(e)}, 400

    def predict_emotion_activity(self, emotion, user_emotions):
        result = {
            "qTable": [
                [5.999999913531018, -9999.0, -9999.0, -9999.0, -9999.0, -9999.0, -9999.0],
                [2.399999897884465, -9999.0, -9999.0, -9999.0, -9999.0, -9999.0, -9999.0], [5.099999914867605, -9999.0, -9999.0, -9999.0, -9999.0, -9999.0, -9999.0], [5.99999950539587, -9999.0, -9999.0, -9999.0, -9999.0, -9999.0, -9999.0], [2.399999742741694, -9999.0, -9999.0, -9999.0, -9999.0, -9999.0, -9999.0], [1.4999999227443992, -9999.0, -9999.0, -9999.0, -9999.0, -9999.0, -9999.0], [5.999999688276037, -9999.0, -9999.0, -9999.0, -9999.0, -9999.0, -9999.0], [5.999999942634708, -9999.0, -9999.0, -9999.0, -9999.0, -9999.0, -9999.0], [3.299999239258464, -9999.0, -9999.0, -9999.0, -9999.0, -9999.0, -9999.0], [2.3999998856026092, -9999.0, -9999.0, -9999.0, -9999.0, -9999.0, -9999.0], [0.5999996541193175, -9999.0, -9999.0, -9999.0, -9999.0, -9999.0, -9999.0], [5.999999893838775, -9999.0, -9999.0, -9999.0, -9999.0, -9999.0, -9999.0], [4.109999323384107, -9999.0, -9999.0, -9999.0, -9999.0, -9999.0, -9999.0], [2.399999404241492, -9999.0, -9999.0, -9999.0, -9999.0, -9999.0, -9999.0], [0.599999867578408, -9999.0, -9999.0, -9999.0, -9999.0, -9999.0, -9999.0], [5.099999785424087, -9999.0, -9999.0, -9999.0, -9999.0, -9999.0, -9999.0], [5.999999999999983, -9999.0, -9999.0, -9999.0, -9999.0, -9999.0, -9999.0], [3.2999989248694384, -9999.0, -9999.0, -9999.0, -9999.0, -9999.0, -9999.0], [2.399999852951441, -9999.0, -9999.0, -9999.0, -9999.0, -9999.0, -9999.0], [5.099999915926454, -9999.0, -9999.0, -9999.0, -9999.0, -9999.0, -9999.0], [0.5999999260046802, -9999.0, -9999.0, -9999.0, -9999.0, -9999.0, -9999.0], [3.2999998635097954, -9999.0, -9999.0, -9999.0, -9999.0, -9999.0, -9999.0], [5.09999845020543, -9999.0, -9999.0, -9999.0, -9999.0, -9999.0, -9999.0], [4.109999513592657, -9999.0, -9999.0, -9999.0, -9999.0, -9999.0, -9999.0], [-9999.0, -3.4999992173020655, -9999.0, -9999.0, -9999.0, -9999.0, -9999.0], [-9999.0, -2.599999899868032, -9999.0, -9999.0, -9999.0, -9999.0, -9999.0], [-9999.0, -4.399999448983412, -9999.0, -9999.0, -9999.0, -9999.0, -9999.0], [-9999.0, 0.09999997580337516, -9999.0, -9999.0, -9999.0, -9999.0, -9999.0], [-9999.0, 0.9999999840315835, -9999.0, -9999.0, -9999.0, -9999.0, -9999.0], [-9999.0, -3.4999994467603344, -9999.0, -9999.0, -9999.0, -9999.0, -9999.0], [-9999.0, -2.5999998938051534, -9999.0, -9999.0, -9999.0, -9999.0, -9999.0], [-9999.0, -0.8899999692531567, -9999.0, -9999.0, -9999.0, -9999.0, -9999.0], [-9999.0, -3.4999998514279893, -9999.0, -9999.0, -9999.0, -9999.0, -9999.0], [-9999.0, 0.999999999999999, -9999.0, -9999.0, -9999.0, -9999.0, -9999.0], [-9999.0, -1.6999999792126446, -9999.0, -9999.0, -9999.0, -9999.0, -9999.0], [-9999.0, -1.6999997505806475, -9999.0, -9999.0, -9999.0, -9999.0, -9999.0], [-9999.0, -4.399999762809235, -9999.0, -9999.0, -9999.0, -9999.0, -9999.0], [-9999.0, -2.599999963740363, -9999.0, -9999.0, -9999.0, -9999.0, -9999.0], [-9999.0, -3.4999999194660574, -9999.0, -9999.0, -9999.0, -9999.0, -9999.0], [-9999.0, 0.09999999084657475, -9999.0, -9999.0, -9999.0, -9999.0, -9999.0], [-9999.0, -1.6999999201170302, -9999.0, -9999.0, -9999.0, -9999.0, -9999.0], [-9999.0, -9999.0, 0.2999999807357683, -9999.0, -9999.0, -9999.0, -9999.0], [-9999.0, -9999.0, 0.2999998950231093, -9999.0, -9999.0, -9999.0, -9999.0], [-9999.0, -9999.0, 0.2999998807164812, -9999.0, -9999.0, -9999.0, -9999.0], [-9999.0, -9999.0, 0.2999999720889883, -9999.0, -9999.0, -9999.0, -9999.0], [-9999.0, -9999.0, -0.6000000224998666, -9999.0, -9999.0, -9999.0, -9999.0], [-9999.0, -9999.0, -2.3999999678427337, -9999.0, -9999.0, -9999.0, -9999.0], [-9999.0, -9999.0, -2.3999999690593676, -9999.0, -9999.0, -9999.0, -9999.0], [-9999.0, -9999.0, 2.9999999999999916, -9999.0, -9999.0, -9999.0, -9999.0], [-9999.0, -9999.0, 2.999999752697935, -9999.0, -9999.0, -9999.0, -9999.0], [-9999.0, -9999.0, -2.3999999642401124, -9999.0, -9999.0, -9999.0, -9999.0], [-9999.0, -9999.0, -0.6000000226152364, -9999.0, -9999.0, -9999.0, -9999.0], [-9999.0, -9999.0, -2.399999941478025, -9999.0, -9999.0, -9999.0, -9999.0], [-9999.0, -9999.0, -1.4999999868235099, -9999.0, -9999.0, -9999.0, -9999.0], [-9999.0, -9999.0, -0.599999929159022, -9999.0, -9999.0, -9999.0, -9999.0], [-9999.0, -9999.0, 0.2999999456728083, -9999.0, -9999.0, -9999.0, -9999.0], [-9999.0, -9999.0, -2.399999931429914, -9999.0, -9999.0, -9999.0, -9999.0], [-9999.0, -9999.0, -0.6000000029853373, -9999.0, -9999.0, -9999.0, -9999.0], [-9999.0, -9999.0, 0.29999999022346335, -9999.0, -9999.0, -9999.0, -9999.0], [-9999.0, -9999.0, -1.4999999229100065, -9999.0, -9999.0, -9999.0, -9999.0], [-9999.0, -9999.0, -1.4999999734708815, -9999.0, -9999.0, -9999.0, -9999.0], [-9999.0, -9999.0, -9999.0, 4.999999866649141, -9999.0, -9999.0, -9999.0], [-9999.0, -9999.0, -9999.0, 0.49999987939870605, -9999.0, -9999.0, -9999.0], [-9999.0, -9999.0, -9999.0, 0.4999999452399735, -9999.0, -9999.0, -9999.0], [-9999.0, -9999.0, -9999.0, 1.3999998409149288, -9999.0, -9999.0, -9999.0], [-9999.0, -9999.0, -9999.0, 0.4999999878763213, -9999.0, -9999.0, -9999.0], [-9999.0, -9999.0, -9999.0, -0.39999997012029304, -9999.0, -9999.0, -9999.0], [-9999.0, -9999.0, -9999.0, 2.299999457566997, -9999.0, -9999.0, -9999.0], [-9999.0, -9999.0, -9999.0, -0.4000000533508316, -9999.0, -9999.0, -9999.0], [-9999.0, -9999.0, -9999.0, 1.399999903398528, -9999.0, -9999.0, -9999.0], [-9999.0, -9999.0, -9999.0, 2.299999637799805, -9999.0, -9999.0, -9999.0], [-9999.0, -9999.0, -9999.0, 1.3999999414266868, -9999.0, -9999.0, -9999.0], [-9999.0, -9999.0, -9999.0, 0.49999993942023246, -9999.0, -9999.0, -9999.0], [-9999.0, -9999.0, -9999.0, 4.99999975321853, -9999.0, -9999.0, -9999.0], [-9999.0, -9999.0, -9999.0, 0.49999981043555597, -9999.0, -9999.0, -9999.0], [-9999.0, -9999.0, -9999.0, 1.3999999609318248, -9999.0, -9999.0, -9999.0], [-9999.0, -9999.0, -9999.0, 4.999999999999988, -9999.0, -9999.0, -9999.0], [-9999.0, -9999.0, -9999.0, -0.4000000455919296, -9999.0, -9999.0, -9999.0], [-9999.0, -9999.0, -9999.0, -9999.0, -0.8999999766111277, -9999.0, -9999.0], [-9999.0, -9999.0, -9999.0, -9999.0, -2.699999490212782, -9999.0, -9999.0], [-9999.0, -9999.0, -9999.0, -9999.0, -3.599999863705301, -9999.0, -9999.0], [-9999.0, -9999.0, -9999.0, -9999.0, -3.599999818235504, -9999.0, -9999.0], [-9999.0, -9999.0, -9999.0, -9999.0, 0.0, -9999.0, -9999.0], [-9999.0, -9999.0, -9999.0, -9999.0, -4.499999356275546, -9999.0, -9999.0], [-9999.0, -9999.0, -9999.0, -9999.0, -5.399997658253229, -9999.0, -9999.0], [-9999.0, -9999.0, -9999.0, -9999.0, -5.399999642421566, -9999.0, -9999.0], [-9999.0, -9999.0, -9999.0, -9999.0, -3.599999793171081, -9999.0, -9999.0], [-9999.0, -9999.0, -9999.0, -9999.0, -2.699999259051761, -9999.0, -9999.0], [-9999.0, -9999.0, -9999.0, -9999.0, -2.6999997828046225, -9999.0, -9999.0], [-9999.0, -9999.0, -9999.0, -9999.0, -2.6999999068556315, -9999.0, -9999.0], [-9999.0, -9999.0, -9999.0, -9999.0, -4.499999783701055, -9999.0, -9999.0], [-9999.0, -9999.0, -9999.0, -9999.0, -0.8999998870802108, -9999.0, -9999.0], [-9999.0, -9999.0, -9999.0, -9999.0, -4.4999997786988395, -9999.0, -9999.0], [-9999.0, -9999.0, -9999.0, -9999.0, 0.0, -9999.0, -9999.0], [-9999.0, -9999.0, -9999.0, -9999.0, -9999.0, 3.9999985405745355, -9999.0], [-9999.0, -9999.0, -9999.0, -9999.0, -9999.0, -0.5000000044778645, -9999.0], [-9999.0, -9999.0, -9999.0, -9999.0, -9999.0, 3.099999806047827, -9999.0], [-9999.0, -9999.0, -9999.0, -9999.0, -9999.0, 3.099999906964918, -9999.0], [-9999.0, -9999.0, -9999.0, -9999.0, -9999.0, 1.2999996987135543, -9999.0], [-9999.0, -9999.0, -9999.0, -9999.0, -9999.0, -1.3999998631522597, -9999.0], [-9999.0, -9999.0, -9999.0, -9999.0, -9999.0, -1.3999999650010564, -9999.0], [-9999.0, -9999.0, -9999.0, -9999.0, -9999.0, 3.999999615412064, -9999.0], [-9999.0, -9999.0, -9999.0, -9999.0, -9999.0, 3.0999997908227632, -9999.0], [-9999.0, -9999.0, -9999.0, -9999.0, -9999.0, 3.999999080197963, -9999.0], [-9999.0, -9999.0, -9999.0, -9999.0, -9999.0, 3.0999999202713973, -9999.0], [-9999.0, -9999.0, -9999.0, -9999.0, -9999.0, -0.5000000041268233, -9999.0], [-9999.0, -9999.0, -9999.0, -9999.0, -9999.0, 3.999999999999996, -9999.0], [-9999.0, -9999.0, -9999.0, -9999.0, -9999.0, -0.5000000141805405, -9999.0], [-9999.0, -9999.0, -9999.0, -9999.0, -9999.0, 3.999998463762669, -9999.0], [-9999.0, -9999.0, -9999.0, -9999.0, -9999.0, -1.3999999645980128, -9999.0], [-9999.0, -9999.0, -9999.0, -9999.0, -9999.0, 3.0999998248995233, -9999.0], [-9999.0, -9999.0, -9999.0, -9999.0, -9999.0, -1.3999998867065284, -9999.0], [-9999.0, -9999.0, -9999.0, -9999.0, -9999.0, 3.09999982514676, -9999.0], [-9999.0, -9999.0, -9999.0, -9999.0, -9999.0, -0.5000000143127104, -9999.0], [-9999.0, -9999.0, -9999.0, -9999.0, -9999.0, 1.2999999641114837, -9999.0], [-9999.0, -9999.0, -9999.0, -9999.0, -9999.0, -9999.0, 1.0999999999999956], [-9999.0, -9999.0, -9999.0, -9999.0, -9999.0, -9999.0, -0.6999999365610712], [-9999.0, -9999.0, -9999.0, -9999.0, -9999.0, -9999.0, 0.10999998582806077], [-9999.0, -9999.0, -9999.0, -9999.0, -9999.0, -9999.0, 0.1099999769345237], [-9999.0, -9999.0, -9999.0, -9999.0, -9999.0, -9999.0, 0.1099999875095698], [-9999.0, -9999.0, -9999.0, -9999.0, -9999.0, -9999.0, -1.5999999919627925], [-9999.0, -9999.0, -9999.0, -9999.0, -9999.0, -9999.0, -1.5999999258400943], [-9999.0, -9999.0, -9999.0, -9999.0, -9999.0, -9999.0, -3.3999998005916097], [-9999.0, -9999.0, -9999.0, -9999.0, -9999.0, -9999.0, -1.599999939027419]]
        }

        activity_ids = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'A16', 'A17', 'A18', 'A19', 'A20', 'A21', 'A22', 'A23', 'A24',
                        'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10', 'B11', 'B12', 'B13', 'B14', 'B15', 'B16', 'B17',
                        'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12', 'F13', 'F14', 'F15', 'F16', 'F17', 'F18', 'F19', 'F20',
                        'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15', 'D16', 'D17',
                        'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8', 'H9', 'H10', 'H11', 'H12', 'H13', 'H14', 'H15', 'H16',
                        'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12', 'S13', 'S14', 'S15', 'S16', 'S17', 'S18', 'S19', 'S20', 'S21',
                        'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9']

        q_table = np.array(result.get('qTable'))
        print(q_table)

        user_emotion_index = user_emotions.index(emotion)
        max_q_valued_column_index = np.argmax(q_table[:, user_emotion_index])
        predicted_activity_id = activity_ids[max_q_valued_column_index]

        print(f"user_emotion_index = {user_emotion_index}")
        print(f"max_q_valued_column_index = {max_q_valued_column_index}")
        print(f"max_q_value = {q_table[max_q_valued_column_index, user_emotion_index]}")
        print(f"predicted_activity_id = {predicted_activity_id}")

        return predicted_activity_id

api.add_resource(facefeature,'/facefeature')
api.add_resource(tonefeature,'/tonefeature')
api.add_resource(textfeature,'/textfeature')
api.add_resource(aggregate,'/aggregate')
api.add_resource(activity,'/activity')

if __name__ == '__main__':
    app.run()