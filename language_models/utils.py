from torch.hub import set_dir
import socket
import numpy as np
import pynvml
import time
import energy_estimator.analyse as simul

if socket.gethostname() == "???":
    userdir = "MODELDIR"
    set_dir(userdir)
else:
    raise "Do not know the directory to load the models from"

import torch
from fairseq.data.data_utils import collate_tokens
from fairseq.models.roberta import RobertaModel
from fairseq.models.transformer import TransformerModel

from wsc import wsc_task, wsc_criterion, wsc_utils
import string
from ga import get_cpu_reading

file_headers = [
        "Epoch", "Mean", "Median",
        "STD", "Top10 Mean", "Top10 STD",
        "Mean Energy", "Top Energy", "Mean Time", "Top Time"
]

def get_characters():
    #return string.ascii_lowercase + string.ascii_uppercase +\
    #" 0123456789!@£$%^&*(){}[]\,./+-)(:"
    #return [chr(x) for x in range(128*10)]
    return [chr(x) for x in range(2048*20)]
    #return [chr(x) for x in range(2048, 2048+1024)]

def get_file(outfile):
    global file_headers
    final = open(outfile, "w")
    final.write(",".join(file_headers)+"\n")
    return final


def get_task(task):
    global userdir

    if task in ["wsc", "mnli"]:
        model = torch.hub.load('pytorch/fairseq', f'roberta.large.{task}',
                user_dir=f"{userdir}")
    elif task == "cola":
        model = RobertaModel.from_pretrained(
            'language_models/checkpoints/',
            checkpoint_file='checkpoint_best.pt',
            data_name_or_path='language_models/CoLA-bin'
        )
    elif task == "wmt18":
        model= TransformerModel.from_pretrained(f'{userdir}/wmt18ensemble/',
                checkpoint_file='wmt18.model1.pt:wmt18.model2.pt:wmt18.model3.pt:wmt18.model4.pt',
                tokenizer='moses', bpe='fastbpe')
    elif task == "wmt19":
        model= TransformerModel.from_pretrained(f'{userdir}/wmt19.en-ru.ensemble/',
                checkpoint_file='model1.pt:model2.pt:model3.pt:model4.pt',
                tokenizer='moses', bpe='fastbpe')
    elif task in ["wmt14", "wmt16"]:
        if task == "wmt16": lang = "en-de"
        elif task == "wmt14": lang = "en-fr"

        model = torch.hub.load('pytorch/fairseq', f'transformer.{task}.{lang}', tokenizer='moses', bpe='subword_nmt', force_reload=False, user_dir=userdir)
    else:
        raise f"I dont task: {task}"

    model.cuda()
    model.eval()

    return model

def write_to_file(_f, epoch, scores, energies, times, ga):

    a = lambda x: np.array(x)

    scores = a(scores)
    energies = a(energies)
    times = a(times)

    mask = (scores != 0)

    scores = scores[mask]
    energies = energies[mask]
    times = times[mask]

    scores_sorted = sorted(scores)
    scores_sorted = scores_sorted[-int(0.1*len(mask)):]

    eng_sorted = [x for _, x in sorted(zip(scores,
            energies), key=lambda x: x[0])]
    eng_sorted = eng_sorted[-int(0.1*len(mask)):]

    times_sorted = [x for _, x in sorted(zip(scores,
            times), key=lambda x: x[0])]
    times_sorted = times_sorted[-int(0.1*len(mask)):]


    f = lambda x: (float(x)/1.0e+9)

    _f.write(
            f"{epoch}, {f(np.mean(scores))}, {f(np.median(scores))},"
            f"{f(np.std(scores))}, {f(np.mean(scores_sorted))},"
            f"{f(np.std(scores_sorted))}, "

            f"{np.mean(energies)}, "
            f"{np.mean(eng_sorted)}, "

            f"{np.mean(times)}, "
            f"{np.mean(times_sorted)}\n"
    )
    _f.flush()

def get_inference(model, task):
    if task == "wsc":
        return lambda x: model.disambiguate_pronoun(x + " [they] ")
    elif task == "mnli":
        return lambda x: model.predict('mnli',
                    mnli_sent_split(model, x)).argmax().item()
    elif task == "cola":
        return lambda x: model.predict('sentence_classification_head',
                    model.encode(x)).argmax().item()
    elif task in ["wmt14", "wmt16", "wmt19", "wmt18"]:
        return lambda x: model.translate(x)
    else:
        raise f"Unknown inference task: {task}"

def mnli_sent_split(model, word):
    if type(word) == tuple:
        sent1, sent2 = word[0], word[1]
    else:
        half1, half2 = word[:len(word)//2], word[len(word)//2:]
        sent1, sent2 = half1[:len(half1)//2] + half2[:len(half2)//2], half1[len(half1)//2:] + half2[len(half2)//2:]
    tokens = model.encode(sent1, sent2)
    return tokens


def get_naturals(pool_size, task, input_size):
    global userdir

    sentences = []
    if task == "wsc":
        for sentence, label in wsc_utils.jsonl_iterator(f'{userdir}/WSC/val.jsonl', eval=True):

            #sentences.append(sentence)
            inx_e = sentence.index("]")
            sentences.append(sentence[inx_e - input_size:inx_e+1])

            if len(sentences) == pool_size:
                break
    elif task == "mnli":
        with open(f'{userdir}/glue_data/MNLI/dev_matched.tsv') as fin:
            fin.readline()
            for index, line in enumerate(fin):
                tokens = line.strip().split('\t')
                sent1, sent2, target = tokens[8], tokens[9], tokens[-1]

                sentences.append((sent1[:input_size//2], sent2[:input_size//2]))

                if len(sentences) == pool_size:
                    break
    elif task == "cola":
        with open(f'{userdir}/glue_data/CoLA/dev.tsv') as fin:
            fin.readline()
            for line in fin:
                tokens = line.strip().split('\t')
                sent = tokens[-1]

                sentences.append(sent[:input_size])

                if len(sentences) == pool_size:
                    break
    elif task.startswith("wmt"):
        with open(f'{userdir}/wmt14data/newsdev2014.en') as fin:
            for line in fin:
                sent = line.strip()
                if len(sent) < input_size:
                    continue

                rnd = np.random.randint(0, len(sent)-input_size)

                sentences.append(sent[rnd:rnd+input_size])

                if len(sentences) == pool_size:
                    break
    else:
        raise f"I cant find naturals for task {task}"

    return sentences


def eval_pass(inference, word, stats, handle, dev="gpu"):
    stats.__reset__()


    if dev == "gpu":
        before_energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
    else:
        before_energy = get_cpu_reading()

    before_time = time.time()

    try:
        prediction = inference(word)
    except Exception as e:
        print(e)
        #Misaligned pronoun here and ValueError
        return 0, 0, None

    after_time = time.time()
    if dev == "gpu":
        after_energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
    else:
        after_energy = get_cpu_reading()

    time_delta   = after_time - before_time
    energy_delta = after_energy - before_energy

    return energy_delta, time_delta, prediction


def hook_up(model, stats, task):
    if task in ["wsc", "cola", "mnli"]:
        stat_hooks = simul.add_hooks(model.model, stats)
    elif task.startswith("wmt"):
        stat_hooks = simul.add_hooks(model.models[0], stats)
    else:
        raise "Not sure how to hook up the model"
    return stat_hooks

def hook_down(stat_hooks):
    simul.remove_hooks(stat_hooks)





