import random
import copy 
import asyncio
from itertools import combinations, product, permutations
from math import exp

from prettytable import PrettyTable
from matplotlib import pyplot as plt

async def get_async_gen(sub): 
    for i in sub:
        yield i

def bool_function(x1: bool, x2: bool, x3: bool, x4: bool) -> bool:
    return not ((x1 and x2) or x3 or x4)

def threshold_function(net):
    return 1 if net >= 0 else 0

def logistic_function(net):
    return 1 / (1 + exp(-net))

def derivative_logistic_function(net):
    return logistic_function(net) * (1 - logistic_function(net))


class SingleLayerNeuralNetwork: 
    def __init__(self, activation_function, derivative, norm=0.3, cutoff=True):
        self.weights = [0, 0, 0, 0, 0]
        self.af = activation_function
        self.der = derivative
        self.norm = norm
        self.cutoff = cutoff

    def _get_network_input(self, inputs):
        return sum(map(lambda x, y: x * y, self.weights, inputs)) + self.weights[-1]

    def _delta_rule(self, error, net, vec):
        vec = list(vec)
        vec.append(1)
        for i in range(len(self.weights)):
            delta = self.norm * error * self.der(net) * vec[i]
            self.weights[i] = self.weights[i] + delta

    def _hamming2(self, s1, s2):
        if len(s1) == len(s2):
            return sum(c1 != c2 for c1, c2 in zip(s1, s2))
        else:
            raise ValueError
    
    def reset_weights(self):
        self.weights = [0, 0, 0, 0, 0]

    def fit(self, training_sample, target, silence=True):
        outs = None
        epochs = 0
        sum_errors = []
        table = PrettyTable()
        table.field_names = ['Epoch', 'Weights', 'Outs', 'Error']
        
        while outs != target and epochs < 5:
            outs = []
            old_weights = copy.copy(self.weights)
            # inds = random.sample(range(len(training_sample)), k=len(training_sample))
            # training_sample = [training_sample[i] for i in inds]
            # target = [target[i] for i in inds]
            for i in range(len(training_sample)):
                net = self.af(self._get_network_input(training_sample[i])) 

                if self.cutoff:
                    y = 1 if net >= 0.5 else 0
                else:
                    y = net

                outs.append(y)
                error = target[i] - y
                self._delta_rule(error, net, training_sample[i])

            serr = self._hamming2(''.join(map(str, outs)), 
                                ''.join(list(map(lambda x: str(int(x)), target))))
            sum_errors.append(serr)
            table.add_row([epochs, list(map(lambda x: round(x, 3), old_weights)), outs, serr])
            epochs += 1

        if not silence:
            print(table)
            plt.plot(list(range(epochs)), sum_errors)
            plt.xlabel("Epoch")
            plt.ylabel("Error")
            plt.title(self.af.__name__)
            plt.grid(True)
            # plt.show()

    def predict(self, sample):
        net = self.af(self._get_network_input(sample))
        if self.cutoff:
            y = 1 if net >= 0.5 else 0
        else:
            y = net
        return y


if __name__ == "__main__":
    sample = [item for item in product([0, 1], repeat=4)]
    main_target = [bool_function(*vec) for vec in sample]

    # nn = SingleLayerNeuralNetwork(threshold_function, lambda _: 1, cutoff=False)
    # nn.fit(sample, main_target, silence=False)
    
    # nn = SingleLayerNeuralNetwork(logistic_function, derivative_logistic_function)
    # nn.fit(sample, main_target, silence=False)

    # nn = SingleLayerNeuralNetwork(threshold_function, lambda _: 1, cutoff=False)
    # best_samples = []
    # not_growing = False
    # for i in range(1, 5):
    #     for comb in combinations(sample, i):
    #         for per in permutations(comb):
    #             nn.reset_weights()
    #             target = [bool_function(*s) for s in per]
    #             nn.fit(per, target)
    #             real_outs = [nn.predict(s) for s in sample]
    #             error = nn._hamming2(''.join(map(str, real_outs)), 
    #                 ''.join(list(map(lambda x: str(int(x)), main_target))))
    #             if error == 0:
    #                 best_samples.append(per)
    # for s in best_samples:
    #     print(s)
    #     nn.reset_weights()
    #     nn.fit(s, [bool_function(*m) for m in s], silence=False)

    nn = SingleLayerNeuralNetwork(logistic_function, derivative_logistic_function)
    best_samples = []
    for i in range(1, 6):
        for comb in combinations(sample, i):
            for per in permutations(comb):
                nn.reset_weights()
                target = [bool_function(*s) for s in per]
                nn.fit(per, target)
                real_outs = [nn.predict(s) for s in sample]
                error = nn._hamming2(''.join(map(str, real_outs)), 
                    ''.join(list(map(lambda x: str(int(x)), main_target))))
                if error == 0:
                    best_samples.append(per)
    for s in best_samples:
        print(s)
        nn.reset_weights()
        nn.fit(s, [bool_function(*m) for m in s], silence=False)
    # s = min(best_samples, key=len)
    # print(s)
    # nn = SingleLayerNeuralNetwork(logistic_function, derivative_logistic_function)
    # nn.fit(s, [bool_function(*m) for m in s], silence=False)

    
