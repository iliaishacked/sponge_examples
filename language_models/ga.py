import numpy as np
from scipy.ndimage import zoom
from scipy import ndimage
from collections import defaultdict

def to_string(dist, dictionary):
    word = " ".join([dictionary[min(int(x*len(dictionary)), len(dictionary)-1)]
        for x in dist])
    return word

def _read_sysfs_file(path):
    with open(path, "r") as f:
        return f.read().strip()

def get_cpu_reading():
    #return int(_read_sysfs_file("/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj"))
    files = [
        int(_read_sysfs_file("/sys/class/powercap/intel-rapl:0/energy_uj")),
        int(_read_sysfs_file("/sys/class/powercap/intel-rapl:1/energy_uj")),
    ]
    return sum(files)

def get_cpu_temp():
    files = [
        int(_read_sysfs_file("/sys/class/thermal/thermal_zone0/temp")),
        int(_read_sysfs_file("/sys/class/thermal/thermal_zone1/temp")),
    ]
    return sum(files)

class GeneticAlgorithm():
    def __init__(self, pool_size, input_shape, clip_min=0, clip_max=1,
            noise_scale=0.1, cv=False):
        self.pool_size = pool_size
        self.input_shape = input_shape
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.noise_scale = noise_scale

        # flag for computer vision
        self.cv = cv
        self._initialise_population()

    def scaled_random_noise(self, shape, scale, dtype):
        noise = np.random.rand(*self.input_shape) * scale
        noise *= self.clip_max - self.clip_min
        noise += self.clip_min
        return noise.astype(dtype)

    def _initialise_population(self):
        self.population = []
        for p in range(self.pool_size):
            samp = np.random.rand(*self.input_shape)
            samp *= self.clip_max - self.clip_min
            samp += self.clip_min
            self.population.append(samp)

    def _random_pick(self, x, axis, p=None):
        axis_index = np.random.choice(x.shape[axis], p=p)
        return x[axis_index, :]

    def selection(self, fitness_scores, best_class=None, perc=None,
            transform=None, rescale=None, rotate=None):

        # top 10% picked
        fitness_sorted = [x for _, x in sorted(zip(fitness_scores,
                self.population), key=lambda x: x[0])]
        scores_sorted = sorted(fitness_scores)

        self.best  = np.dstack(fitness_sorted[-10:])
        self.worst = np.dstack(fitness_sorted[:10])

        percentile90 = int(len(fitness_sorted) * 0.1)

        top10 = fitness_sorted[-percentile90:]
        p     = scores_sorted[-percentile90:]

        if self.cv:
            population = []
        else:
            #population = top10[:]
            population = top10[:]

        parents    = top10[:]

        if best_class is not None:
            best_class = np.array(best_class)
            best = defaultdict(lambda: (0,0))
            for i, (cl, sc) in enumerate(zip(best_class, fitness_scores)):
                if best[cl][0] < sc:
                    best[cl] = (sc, i)

            for key in best:
                parents.append(self.population[best[key][1]])
                p.append(best[key][0])

                population.append(self.population[best[key][1]])

        parents = np.array(parents)
        p = np.array(p)
        p = p / p.sum()

        assert len(p) == len(parents)
        pop_list = [x.tolist() for x in population]

        while (len(population) < self.pool_size):
            # heuristic, pure cross over
            parent_a = self._random_pick(parents, axis=0, p=p)
            parent_b = self._random_pick(parents, axis=0, p=p)
            mutated = [
                self.crossover(parent_a, parent_b),
                self.crossover(parent_b, parent_a)
            ]

            for mut in mutated:
                mut = self.mutate(mut, perc=perc)
                mut = self.clip(mut)
                mutated_list = mut.tolist()
                if mutated_list not in pop_list:
                    pop_list.append(mutated_list)
                    population.append(mut)

        del pop_list
        del self.population
        np.random.shuffle(population)
        self.population = population
        self.top10 = sorted(fitness_scores)[-percentile90:]

    def mutate(self, a, perc=None, transform=None, rescale=None,
            rotate=None):

        a_shape = a.shape

        if rotate is not None:
            a_rot = ndimage.rotate(a, np.random.random()*360, axes=(2,3), reshape=False)
            a += a_rot

        if rescale is not None:
            if np.random.random() < rescale:
                new_width = max(0.1, np.random.random())
                new_height= max(0.1, np.random.random())

                a_original = a
                a = zoom(a, (1, 1,
                    new_width, new_height,))

                a = np.pad(a,
                    (
                        (0,0),
                        (0,0),
                        (0, a_shape[2]-a.shape[2]),
                        (0, a_shape[3]-a.shape[3])),
                    mode="constant", constant_values=(self.clip_min,),
                    )
                #a += a_original

        if transform is not None:
            for ax in range(1, len(a_shape)):
                if np.random.random() < transform:
                    a = np.roll(a, shift=np.random.randint(low=0, high=a_shape[ax]), axis=ax)

            if np.random.random() < transform:
                a = np.transpose(a, (0,1,3,2))

        if perc is not None:

            rperc = perc#np.random.rand() * perc

            indx = (np.random.random(size=int(rperc*a.size)) *
                    a.size).astype(int)
            values = self.scaled_random_noise(indx.shape,
                scale=self.noise_scale, dtype=a.dtype)
            a = a.flatten()

            if not self.cv:
                if np.random.rand() < 0.5:
                    a = np.flip(a)

            np.put(a, indx, values)
            a = a.reshape(a_shape)
            return a
        else:
            if not self.cv:
                if np.random.rand() < 0.5:
                    a = np.flip(a)

            noise = self.scaled_random_noise(a.shape,
                scale=self.noise_scale, dtype=a.dtype)
            noise -= self.noise_scale*float(self.clip_max - self.clip_min)/2
            return a + noise

    def crossover(self, a, b, concat=False):
        a_shape = a.shape
        flat_a , flat_b = a.flatten(), b.flatten()
        mid_point = len(flat_a) // 2

        #offspring = sbx(
        #        flat_a, flat_b,
        #        -2.8, 2.8,
        #        eta=1
        #        )
        #offspring = offspring.reshape(a_shape)
        #return offspring

        if concat:
            left = flat_a[:mid_point]
            right = flat_b[mid_point:]
            offspring = np.concatenate((left, right), axis=0)
        else:
            masked = np.random.rand(*flat_a.shape)
            offspring = (flat_a * masked + flat_b *(1-masked))
        offspring = offspring.reshape(a_shape)
        return offspring

    def clip(self, a):
        return np.clip(a, self.clip_min, self.clip_max)

def test():
    pool_size = 100
    input_shape = (32, 32, 3)
    ga = GeneticAlgorithm(pool_size, input_shape)
    print(ga.population)
    ga.selection(list(np.random.rand(pool_size)))
    print(ga.population)



def sbx(x, y, _min, _max, eta):
    '''
    SBX (cf Deb 2001, p 113) Simulated Binary Crossover
    A large value ef eta gives a higher probablitity for
    creating a `near-parent' solutions and a small value allows
    distant solutions to be selected as offspring.
    '''
    xl = _min
    xu = _max
    z = x.copy()
    r1 = np.random.random(size=len(x))
    r2 = np.random.random(size=len(x))

    for i in range(0, len(x)):
        if abs(x[i] - y[i]) > 1e-15:
            x1 = min(x[i], y[i])
            x2 = max(x[i], y[i])

            beta = 1.0 + (2.0 * (x1 - xl) / (x2 - x1))
            alpha = 2.0 - beta ** -(eta + 1)
            rand = r1[i]
            if rand <= 1.0 / alpha:
                beta_q = (rand * alpha) ** (1.0 / (eta + 1))
            else:
                beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1))

            c1 = 0.5 * (x1 + x2 - beta_q * (x2 - x1))

            beta = 1.0 + (2.0 * (xu - x2) / (x2 - x1))
            alpha = 2.0 - beta ** -(eta + 1)
            if rand <= 1.0 / alpha:
                beta_q = (rand * alpha) ** (1.0 / (eta + 1))
            else:
                beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1))
            c2 = 0.5 * (x1 + x2 + beta_q * (x2 - x1))

            c1 = min(max(c1, xl), xu)
            c2 = min(max(c2, xl), xu)

            if r2[i] <= 0.5:
                z[i] = c2
            else:
                z[i] = c1
    return z

#test()
