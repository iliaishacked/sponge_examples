import torch
from subprocess import Popen
import signal
import time

'''
English to German translation
download model from
https://github.com/pytorch/fairseq/tree/master/examples/wmt19
'''
# List available models

from torch.hub import set_dir
import socket

if socket.gethostname() == "idun.cl.cam.ac.uk":
    set_dir("/local/scratch-3/yaz21/ilia/")

# Load a transformer trained on WMT'16 En-De
en2de = torch.hub.load('pytorch/fairseq', 'transformer.wmt16.en-de', tokenizer='moses', bpe='subword_nmt', force_reload=False)
en2de.eval()  # disable dropout

# The underlying model is available under the *models* attribute
#assert isinstance(en2de.models[0], fairseq.models.transformer.TransformerModel)

# Move model to GPU for faster translation
en2de.cuda()

phrases = [
    "Aaron is beautiful",
    " ",
    "Aaron is very beautiful",
    "HELLO A",
    "                 ",
    "00000000000000000",
    "A",
    "Aaron is very gorgeous",
    "Go away you nasty bastard!",
    "hello a",
    "hello b",
    ]

hot_start = 30
for _ in range(hot_start):
    print(en2de.translate("JAJAJAJJAJAJ"))

# Translate a sentence
for i in range(20):
    for j in range(len(phrases)):
        p1 = Popen(["./../prog", f"reading/{i}_{j}.csv", "\"{phrases[j]}\""])
        time.sleep(2)
        kk = en2de.translate(phrases[j])
        time.sleep(2)
        p1.send_signal(signal.SIGINT)
        print(kk)
        time.sleep(1)
        p1.wait()

