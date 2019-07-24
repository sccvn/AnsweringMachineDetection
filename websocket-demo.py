#!/usr/bin/env python

from __future__ import absolute_import, print_function
import glob
import wave
import random
import struct
import datetime
import argparse
import io
import logging
import os
import sys
import time
from logging import debug, info
import uuid
import cgi
import audioop
import requests
import tornado.ioloop
import tornado.websocket
import tornado.httpserver
import tornado.template
import tornado.web
import webrtcvad
from tornado.web import url
import json
from base64 import b64decode
import nexmo
import collections
import pickle
import librosa
import numpy as np
import argparse


from dotenv import load_dotenv
load_dotenv()

# Only used for record function

logging.captureWarnings(True)

# Constants:
MS_PER_FRAME = 20  # Duration of a frame in ms
RATE = 16000
SILENCE = 10 # How many continuous frames of silence determine the end of a phrase
CLIP_MIN_MS = 200  # ms - the minimum audio clip that will be used
MAX_LENGTH = 3000  # Max length of a sound clip for processing in ms
VAD_SENSITIVITY = 3
CLIP_MIN_FRAMES = CLIP_MIN_MS // MS_PER_FRAME

# Global variables
conns = {}
conversation_uuids = collections.defaultdict(list)
nexmo_client = None
loaded_model = pickle.load(open("models/GaussianProcessClassifier-20190724T1739.pkl", "rb"))
print(loaded_model)


# Environment Variables, these are set in .env locally
HOSTNAME = os.getenv("HOSTNAME")
PORT = os.getenv("PORT")

MY_LVN = os.getenv("MY_LVN")
APP_ID = os.getenv("APP_ID")
PROJECT_ID = os.getenv("PROJECT_ID")
CLOUD_STORAGE_BUCKET = os.getenv("CLOUD_STORAGE_BUCKET")

ANSWERING_MACHINE_TEXT = os.getenv("ANSWERING_MACHINE_TEXT")

parser = argparse.ArgumentParser()
parser.add_argument("debug",nargs='?')
args = parser.parse_args()

from google.cloud import storage
storage_client = storage.Client()
bucket = storage_client.get_bucket(CLOUD_STORAGE_BUCKET)

def _get_private_key():
    try:
        return os.environ['PRIVATE_KEY']
    except:
        with open('private.key', 'r') as f:
            private_key = f.read()

    return private_key

PRIVATE_KEY = _get_private_key()

if args.debug:
    with wave.open("wav_stream.wav", 'wb') as wav_out:
        for wav_path in glob.glob("test_files/*.wav"):
            with wave.open(wav_path, 'rb') as wav_in:
                if not wav_out.getnframes():
                    wav_out.setparams(wav_in.getparams())
                for i in range(20000):
                   value = random.randint(0, 1)
                   data = struct.pack('<h', value)
                   wav_out.writeframesraw( data )
                wav_out.writeframes(wav_in.readframes(wav_in.getnframes()))

def debugNCCO(request,conversation_uuid):
    ncco = [
          {
             "action": "connect",
             # "eventUrl": [self.request.protocol +"://" + self.request.host  +"/event"],
             "from": MY_LVN,
             "endpoint": [
                 {
                    "type": "websocket",
                    "uri" : "ws://"+request.host +"/socket",
                    "content-type": "audio/l16;rate=16000",
                    "headers": {
                        "conversation_uuid":conversation_uuid #change to user
                    }
                 }
             ]
           },
           {
            "action": "stream",
            "streamUrl": ["https://"+request.host+"/static/wav_stream.wav"]
          }
        ]
    return ncco

class BufferedPipe(object):
    def __init__(self, max_frames, sink):
        """
        Create a buffer which will call the provided `sink` when full.

        It will call `sink` with the number of frames and the accumulated bytes when it reaches
        `max_buffer_size` frames.
        """
        self.sink = sink
        self.max_frames = max_frames

        self.count = 0
        self.payload = b''

    def append(self, data, id):
        """ Add another data to the buffer. `data` should be a `bytes` object. """

        self.count += 1
        self.payload += data

        if self.count == self.max_frames:
            self.process(id)

    def process(self, id):
        """ Process and clear the buffer. """
        self.sink(self.count, self.payload, id)
        self.count = 0
        self.payload = b''

class NexmoClient(object):
    def __init__(self):
        self.client = nexmo.Client(application_id=APP_ID, private_key=PRIVATE_KEY)

    def hangup(self,conversation_uuid):
        for event in conversation_uuids[conversation_uuid]:
            try:
                response = self.client.update_call(event["uuid"], action='hangup')
                print("hangup uuid {} response: {}".format(event["conversation_uuid"], response))
            except Exception as e:
                print("Hangup error",e)
        conversation_uuids[conversation_uuid].clear()

    def speak(self, conversation_uuid):
        uuids = [event["uuid"] for event in conversation_uuids[conversation_uuid] if event["from"] == MY_LVN and "ws" not in event["to"]]
        uuid = next(iter(uuids), None)
        print(uuid)
        if uuid is not None:
            print('found {}'.format(uuid))
            response = self.client.send_speech(uuid, text=ANSWERING_MACHINE_TEXT)
            print("send_speech response",response)
        else:
            print("{} does not exist in list {}".format(conversation_uuid, conversation_uuids[conversation_uuid]))

class AudioProcessor(object):
    def __init__(self, path, conversation_uuid):
        self._path = path
        self.conversation_uuid = conversation_uuid

    def process(self, count, payload, conversation_uuid):
        if count > CLIP_MIN_FRAMES :  # If the buffer is less than CLIP_MIN_MS, ignore it
            print("record clip")
            fn = "rec-{}-{}.wav".format(conversation_uuid,datetime.datetime.now().strftime("%Y%m%dT%H%M%S"))
            output = wave.open(fn, 'wb')
            output.setparams(
                (1, 2, RATE, 0, 'NONE', 'not compressed'))
            output.writeframes(payload)
            output.close()
            prediction = self.predict_from_file(fn)
            print("prediction",prediction)

            if args.debug == None:
                self.upload_to_gcp(fn, conversation_uuid)
            self.remove_file(fn)

            if prediction[0] == 0:
                nexmo_client.speak(conversation_uuid)
                if args.debug == None:
                    time.sleep(2)
                    nexmo_client.hangup(conversation_uuid)
        else:
            info('Discarding {} frames'.format(str(count)))

    def predict_from_file(self, wav_file):
        if loaded_model != None:
            print("load file {}".format(wav_file))
            start = time.time()
            X, sample_rate = librosa.load(wav_file, res_type='kaiser_fast')
            # X = librosa.resample(X, sample_rate, 16000, res_type='kaiser_fast')
            mfccs_40 = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
            prediction = loaded_model.predict([mfccs_40])

            # stft =dfd np.abs(librosa.stft(X))
            # mfccs_40 = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
            # chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            # mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate,n_mels=128,fmax=8000).T,axis=0)
            # contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
            # tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),
            # sr=sample_rate).T,axis=0)
            #
            # a = [mfccs_40, chroma, mel, contrast, tonnetz]
            #
            # total_len = 0
            # for f in a:
            #   total_len += f.shape[0]
            #
            # features = np.vstack([np.empty((0,total_len)),np.hstack(a)])
            # prediction = loaded_model.predict(features)
            # predict_proba = loaded_model.predict_proba(features)

            # print("prediction",prediction)
            # print("predict_proba",predict_proba)

            end = time.time()
            print ("time {}".format(end - start))
            return  prediction

    def remove_file(self, wav_file):
        os.remove(wav_file)

    def upload_to_gcp(self, wav_file, conversation_uuid):
        data = conversation_uuids[conversation_uuid][0]
        if PROJECT_ID and CLOUD_STORAGE_BUCKET:
            blob = bucket.blob("split_recordings/"+data["from"]+"_"+conversation_uuid + "/"+wav_file)
            blob.upload_from_filename(wav_file, content_type="audio/wav")
            print('File uploaded.')




class WSHandler(tornado.websocket.WebSocketHandler):
    def initialize(self):
        # Create a buffer which will call `process` when it is full:
        self.frame_buffer = None
        # Setup the Voice Activity Detector
        self.tick = None
        self.id = None#uuid.uuid4().hex
        self.vad = webrtcvad.Vad()
        # Level of sensitivity

        self.vad.set_mode(VAD_SENSITIVITY)

        self.processor = None
        self.path = None
        self.nexmo_client = NexmoClient()


    def open(self, path):
        info("client connected")
        debug(self.request.uri)
        self.path = self.request.uri
        self.tick = 0

    def on_message(self, message):
        # Check if message is Binary or Text
        if type(message) != str:
            if self.vad.is_speech(message, RATE):
                debug("SPEECH from {}".format(self.id))
                self.tick = SILENCE
                self.frame_buffer.append(message, self.id)
            else:
                debug("Silence from {} TICK: {}".format(self.id, self.tick))
                self.tick -= 1
                if self.tick == 0:
                    # Force processing and clearing of the buffer
                    self.frame_buffer.process(self.id)
        else:
            info(message)
            # Here we should be extracting the meta data that was sent and attaching it to the connection object
            data = json.loads(message)

            if data.get('content-type'):
                conversation_uuid = data.get('conversation_uuid') #change to use
                self.id = conversation_uuid
                conns[self.id] = self
                self.processor = AudioProcessor(
                    self.path, conversation_uuid).process
                self.frame_buffer = BufferedPipe(MAX_LENGTH // MS_PER_FRAME, self.processor)
                self.write_message('ok')

    def on_close(self):
        # Remove the connection from the list of connections
        del conns[self.id]
        print("client disconnected")

class EventHandler(tornado.web.RequestHandler):
    @tornado.web.asynchronous
    def post(self):
        data = json.loads(self.request.body)
        if data["status"] == "answered":
            print(data)

            conversation_uuid = data["conversation_uuid"]
            conversation_uuids[conversation_uuid].append(data)

        if data["to"] == MY_LVN and data["status"] == "completed":
            conversation_uuid = data["conversation_uuid"]
            nexmo_client.hangup(conversation_uuid)
        self.content_type = 'text/plain'
        self.write('ok')
        self.finish()

class EnterPhoneNumberHandler(tornado.web.RequestHandler):
    @tornado.web.asynchronous
    def get(self):
        print(self.request)
        ncco = []
        if args.debug:
            ncco = debugNCCO(self.request,self.get_argument("conversation_uuid"))
        else:
            ncco = [
                  {
                    "action": "talk",
                    "text": "Please enter a phone number to dial"
                  },
                  {
                    "action": "input",
                    "eventUrl": [self.request.protocol +"://" + self.request.host +"/ivr"],
                    "timeOut":10,
                    "maxDigits":12,
                    "submitOnHash":True
                  }

                ]
        self.write(json.dumps(ncco))
        self.set_header("Content-Type", 'application/json; charset="utf-8"')
        self.finish()


class AcceptNumberHandler(tornado.web.RequestHandler):
    @tornado.web.asynchronous
    def post(self):
        data = json.loads(self.request.body)
        print(data)
        ncco = [
              {
                "action": "talk",
                "text": "Thanks. Connecting you now"
              },
             {
             "action": "connect",
              # "eventUrl": [self.request.protocol +"://" + self.request.host  + "/event"],
               "from": MY_LVN,
               "endpoint": [
                 {
                   "type": "phone",
                   "number": data["dtmf"]
                 }
               ]
             },
              {
                 "action": "connect",
                 # "eventUrl": [self.request.protocol +"://" + self.request.host  +"/event"],
                 "from": MY_LVN,
                 "endpoint": [
                     {
                        "type": "websocket",
                        "uri" : "ws://"+self.request.host +"/socket",
                        "content-type": "audio/l16;rate=16000",
                        "headers": {
                            "conversation_uuid":data["conversation_uuid"] #change to user
                        }
                     }
                 ]
               }
            ]
        self.write(json.dumps(ncco))
        self.set_header("Content-Type", 'application/json; charset="utf-8"')
        self.finish()

class RecordHandler(tornado.web.RequestHandler):
    @tornado.web.asynchronous
    def post(self):
        data = json.loads(self.request.body)

        response = client.get_recording(data["recording_url"])
        fn = "call-{}.wav".format(data["conversation_uuid"])

        if PROJECT_ID and CLOUD_STORAGE_BUCKET:
            blob = bucket.blob(fn)
            blob.upload_from_string(response, content_type="audio/wav")
            print('File uploaded.')

        self.write('ok')
        self.set_header("Content-Type", 'text/plain')
        self.finish()

class PingHandler(tornado.web.RequestHandler):
    @tornado.web.asynchronous
    def get(self):
        self.write('ok')
        self.set_header("Content-Type", 'text/plain')
        self.finish()


def main():
    try:
        global nexmo_client
        nexmo_client = NexmoClient()

        logging.basicConfig(
            level=logging.INFO,
            format="%(levelname)7s %(message)s",
        )
        application = tornado.web.Application([
			url(r"/ping", PingHandler),
            (r"/static/(.*)", tornado.web.StaticFileHandler, {"path": ""}),
            (r"/event", EventHandler),
            (r"/ncco", EnterPhoneNumberHandler),
            (r"/recording", RecordHandler),
            (r"/ivr", AcceptNumberHandler),
            url(r"/(.*)", WSHandler),
        ])
        http_server = tornado.httpserver.HTTPServer(application)
        port = int(os.getenv('PORT', 8000))
        http_server.listen(port)
        tornado.ioloop.IOLoop.instance().start()
    except KeyboardInterrupt:
        pass  # Suppress the stack-trace on quit


if __name__ == "__main__":
    main()
