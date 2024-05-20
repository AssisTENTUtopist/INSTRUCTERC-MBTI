import argparse
import queue
import sys
import json
import sounddevice as sd

from vosk import Model, KaldiRecognizer, SetLogLevel

SetLogLevel(-1)
q = queue.Queue()

def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text

def callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status, file=sys.stderr)
    q.put(bytes(indata))

def stt():
    filename="prompt.txt"
    model=Model(lang="ru")
    print('Произнесите текст и прервите промпт с помощью ^C.')
    device=None
    device_info = sd.query_devices(device, "input")
    samplerate = int(device_info["default_samplerate"])

    dump_fn = open(filename, "w")
    result=""
    try:
        with sd.RawInputStream(samplerate=samplerate, blocksize = 8000, device=device,
                dtype="int16", channels=1, callback=callback):

            rec = KaldiRecognizer(model, samplerate)
            while True:
                data = q.get()
                if rec.AcceptWaveform(data):
                    res = json.loads(rec.Result())
                    if res["text"]:
                        #print(res["text"], end="")
                        result+=res["text"]
                        if dump_fn is not None:
                            dump_fn.write(res["text"]+' ')

    except KeyboardInterrupt:
        res = json.loads(rec.FinalResult())
        #print(res["text"])
        result+=res["text"]
        if dump_fn is not None:
            dump_fn.write(res["text"]+' ')
        return result
