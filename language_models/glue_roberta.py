import torch

from torch.hub import set_dir
import socket

if socket.gethostname() == "idun.cl.cam.ac.uk":
    set_dir("/local/scratch-3/yaz21/ilia/")

from fairseq.data.data_utils import collate_tokens

roberta = torch.hub.load('pytorch/fairseq', 'roberta.large.mnli')

label_map = {0: 'contradiction', 1: 'neutral', 2: 'entailment'}
ncorrect, nsamples = 0, 0
roberta.cuda()
roberta.eval()
with open('glue_data/MNLI/dev_matched.tsv') as fin:
    fin.readline()
    for index, line in enumerate(fin):
        tokens = line.strip().split('\t')
        sent1, sent2, target = tokens[8], tokens[9], tokens[-1]
        tokens = roberta.encode(sent1, sent2)
        prediction = roberta.predict('mnli', tokens).argmax().item()
        prediction_label = label_map[prediction]
        ncorrect += int(prediction_label == target)
        nsamples += 1
print('| Accuracy: ', float(ncorrect)/float(nsamples))
