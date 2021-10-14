import click
import pickle
import time
import numpy as np
import pynvml
import torch
from language_models.utils import\
    (get_characters, get_inference, get_file,
     get_task, write_to_file, get_naturals, eval_pass, hook_up, hook_down)
from ga import GeneticAlgorithm, to_string
import energy_estimator.analyse as simul
from collections import defaultdict
import requests, uuid


@click.group()
@click.version_option()
def cli():
    """ Salo -- energy attacks against DNNs """

@click.command(help="Command to run the analysis")
@click.argument('task')
@click.option('pool_size', '--pool_size', default=50)
@click.option('input_size', '--input_size', default=6)
@click.option('epochs', '--epochs', default=50)
@click.option('out', '--out', default="test")
@click.option('hot_start', '--hot_start', is_flag=True)
@click.option('natural', '--natural', is_flag=True)
@click.option('cost_type', '--cost_type', default="simul",
        type=click.Choice(['simul', 'time', 'energy']))
@click.option('nointereng', '--nointereng', is_flag=True)
def analyse(task, pool_size, input_size, epochs, out,
        hot_start,natural, cost_type, nointereng):
    ga = GeneticAlgorithm(pool_size, (input_size,), noise_scale=1.0)
    hardware = simul.HardwareModel(optim=True)

    dictionary = get_characters()
    _f = get_file(out)

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    stats = simul.StatsRecorder()

    model = get_task(task)
    inference = get_inference(model, task)

    if hot_start:
        with torch.no_grad():
            for i in range(30):
                try:
                    inference("JAJJAJAJAJJAJAJAJAJJ")
                except:
                    pass

    all_scores, all_energies, all_simuls, all_times = [], [], [], []

    if natural:
        natural_samples = get_naturals(pool_size, task, input_size)
        scores,energies,times,simuls = [],[],[],[]

        with torch.no_grad():
            for i, word in enumerate(natural_samples):
                energy_delta, time_delta, prediction =\
                    eval_pass(inference, word, stats, handle)

                stat_hooks = hook_up(model, stats, task)
                eval_pass(inference, word, stats, handle)
                hook_down(stat_hooks)

                if energy_delta == 0:
                    energy_est = 0
                else:
                    energy_est = simul.get_energy_estimate(stats, hardware)

                simuls.append(energy_est)
                energies.append(energy_delta)
                times.append(time_delta)
                scores.append(-1)

        all_simuls.append(simuls)
        all_energies.append(energies)
        all_times.append(times)

        write_to_file(_f, -1, simuls, energies, times, ga)
        del natural_samples

    for epoch in range(epochs):
        scores,energies,simuls,times = [],[],[],[]

        for i, sample in enumerate(ga.population):
            word = to_string(sample, dictionary)
            with torch.no_grad():
                energy_delta, time_delta, prediction =\
                    eval_pass(inference, word, stats, handle)

                if not nointereng or (epoch == epochs-1):
                    stat_hooks = hook_up(model, stats, task)
                    eval_pass(inference, word, stats, handle)
                    hook_down(stat_hooks)

                if energy_delta == 0:
                    energy_est = 0
                else:
                    if nointereng and (epoch != epochs-1):
                        energy_est = -2
                    else:
                        energy_est = simul.get_energy_estimate(stats, hardware)

                if cost_type == "energy":
                    scores.append(energy_delta)
                elif cost_type == "time":
                    scores.append(time_delta)
                elif cost_type == "simul":
                    scores.append(energy_est)
                else:
                    raise "Unsupported cost type {cost_type}"

                simuls.append(energy_est)
                energies.append(energy_delta)
                times.append(time_delta)

                print(f"{word} -> {prediction} : Simul: {energy_est:.4f} Time:"
                      f" {time_delta:.4f} NVML: {energy_delta:.4f}", end="\r")

        ga.selection(scores, perc=0.1)
        write_to_file(_f, epoch, scores, energies, times, ga)

        all_simuls.append(simuls)
        all_energies.append(energies)
        all_times.append(times)

        print()
        print("Best:", to_string(ga.best[0][:,-1], dictionary))
        print("Worst:", to_string(ga.worst[0][:,-1], dictionary))
        print(f"Epoch: {epoch}/{epochs}: "
              f"Mean simul: {np.mean(simuls)} {np.max(simuls)} "
              f"Mean nvml: {np.mean(energies)} {np.max(energies)} "
              f"Mean time: {np.mean(times)} {np.max(times)}")
        #del scores, energies, times
        del ga.top10

    _f.close()

    with open(out+".pkl", "wb") as _f:
        pickle.dump([
            task, cost_type, ga.population,
            all_simuls, all_energies, all_times], _f)

@click.command(help="Command to run the analysis")
@click.argument('task')
@click.option('infile', '--infile', required=True)
@click.option('out', '--out', default="test")
@click.option('natural', '--natural', is_flag=True)
@click.option('natural_size', '--natural_size', default=None, type=int)
def run(task, infile, out, natural, natural_size):

    inf = pickle.load(open(infile, "rb"))

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    stats = simul.StatsRecorder()
    hardware = simul.HardwareModel(optim=True)

    model = get_task(task)
    dictionary = get_characters()

    inference = get_inference(model, task)

    energies,times,simuls = defaultdict(list), defaultdict(list), defaultdict(list)
    for dev in ["cpu", "gpu"]:

        if dev == "cpu":
            model = model.to("cpu")
        else:
            model = model.cuda(0)

        if natural:
            samples = get_naturals(100, task, natural_size)
        else:
            if (len(inf) == 6) and (type(inf[0]) == str):
                samples = inf[2]
            else:
                samples = inf
            np.random.shuffle(samples)
            #samples = samples[::-1]

        for i, sample in enumerate(samples):
            if i == 100:
                break
            if natural:
                word = sample
            else:
                word = to_string(sample, dictionary)

            with torch.no_grad():
                energy_delta, time_delta, prediction =\
                    eval_pass(inference, word, stats, handle, dev=dev)

                stat_hooks = hook_up(model, stats, task)
                eval_pass(inference, word, stats, handle)
                hook_down(stat_hooks)

                if energy_delta == 0:
                    energy_est = 0
                else:
                    energy_est = simul.get_energy_estimate(stats, hardware)

                simuls[dev].append(energy_est)
                energies[dev].append(energy_delta)
                times[dev].append(time_delta)

                print(f"{word} -> {prediction} : Simul: {energy_est:.4f} Time:"
                      f" {time_delta:.4f} NVML: {energy_delta:.4f}", end="\r")

    with open(out+".pkl", "wb") as _f:
        pickle.dump([
            task, inf,
            simuls, energies, times], _f)



@click.command(help="Command to run the azure exps")
@click.argument('task')
@click.option('pool_size', '--pool_size', default=50)
@click.option('input_size', '--input_size', default=6)
@click.option('epochs', '--epochs', default=50)
@click.option('out', '--out', default="test")
def azure(task, pool_size, input_size, epochs, out):
    ga = GeneticAlgorithm(pool_size, (input_size,), noise_scale=1.0)
    hardware = simul.HardwareModel(optim=True)

    dictionary = get_characters()
    _f = get_file(out)

    subscription_key = ""
    endpoint = "https://somethingsomething.cognitiveservices.azure.com/"

    if task == "translate":
        subscription_key = ""
        endpoint = "https://api.cognitive.microsofttranslator.com/"

    if task == "language_detection":
        language_api_url = endpoint + "/text/analytics/v3.0/languages"
    elif task == "sentiment":
        language_api_url = endpoint + "/text/analytics/v3.0/sentiment"
    elif task == "keyphrase":
        language_api_url = endpoint + "/text/analytics/v3.0/keyPhrases"
    elif task == "indetities":
        language_api_url = endpoint + "/text/analytics/v3.0/entities/recognition/general"
    elif task == "translate":
        language_api_url = endpoint + "/translate?api-version=3.0"
    else:
        raise "I do not know such a task"

    all_scores = []
    for epoch in range(epochs):
        scores = []

        for i, sample in enumerate(ga.population):
            word = to_string(sample, dictionary)
            documents = {"documents": [{"id": "1", "language": "en", "text": word}]}

            if task == "translate":
                documents = [{"Text":word}]

                headers = {
                        "Ocp-Apim-Subscription-Key": subscription_key,
                        "Ocp-Apim-Subscription-Region": "westeurope",
                }
            else:
                headers = {"Ocp-Apim-Subscription-Key": subscription_key}

            response  = requests.post(language_api_url + "&to=fr", headers=headers, json=documents)
            timedelta = response.elapsed.total_seconds()
            languages = response.json()

            scores.append(timedelta)

            if 'documents' not in languages:
                print(f"{word[:10]} -> ERROR ({languages}) : {timedelta:.3f}", end='\r')
            else:
                #print(f"{word[:10]} -> {languages['documents'][0]} : {timedelta:.3f}", end='\r')
                print(f"{word[:10]} -> {timedelta:.3f}", end='\r')

        ga.selection(scores, perc=0.1)
        all_scores.append(scores)

        print()
        print("Best:", to_string(ga.best[0][:,-1], dictionary))
        print("Worst:", to_string(ga.worst[0][:,-1], dictionary))
        print(f"Epoch: {epoch}/{epochs}: "
                f"Mean time: {np.mean(scores)} 90perc: {np.percentile(scores, 90)} max {np.max(scores)}")
        #del scores, energies, times
        del ga.top10

    with open(out+".pkl", "wb") as _f:
        pickle.dump([
            task, ga.population, all_scores, scores], _f)


cli.add_command(run)
cli.add_command(analyse)
cli.add_command(azure)

if __name__ == '__main__':
    cli()
