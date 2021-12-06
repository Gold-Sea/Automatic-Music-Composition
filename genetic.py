import json
import numpy as np
from numpy.lib.function_base import append, select
import model
import IO
import fitness

def takeSecond(elem):
    return elem[1]

with open("./GA.json",'r') as load_f:
    load_dict = json.load(load_f)
    num_tones = load_dict['num_tones']
    iters = load_dict['iters']
    num_pop = load_dict['num_pop']


class Chromosome:
    def __init__(self, jsonstr, genes = None):
        assert isinstance(jsonstr, dict)

        num_tone = jsonstr['num_tones']
        self.length = jsonstr['bars'] * jsonstr['pulses']
        self.num_tone = num_tone
        self.geneset = [i for i in range(self.num_tone)]
        self.genes = genes
    
    # return a numpy array with shape(num_tones + 2,)
    def generate_parent(self): 
        self.genes = np.random.randint(low=0, high=self.num_tone,size=(self.length,))

    def register(self, func):
        assert isinstance(func, function)
        self.Fitness = func

    def mutate(self, typeid):
        genes = self.genes.copy()
        if typeid == 0:
            randindex = np.random.randint(low=0, high=self.length)
            if_pluse = np.random.randint(low=0, high=2)
            original_val = genes[randindex]
            # Upward overflow
            if original_val + 8 >= self.num_tone:
                if_pluse = 0
            if original_val - 8 < 0:
                if_pluse = 1
            if if_pluse:
                genes[randindex] += 8
            else:
                genes[randindex] -= 8

        elif typeid == 1:
            randindex = np.random.randint(low=0, high=self.length)
            original_val = genes[randindex]
            randval = np.random.randint(low=0, high=self.num_tone)
            while(randval == original_val):
                randval = np.random.randint(low=0, high=self.num_tone)
            genes[randindex] = randval

        elif typeid == 2:
            randindex = np.random.randint(low=0, high=self.length - 1)
            temp = genes[randindex]
            genes[randindex] = genes[randindex + 1]
            genes[randindex + 1] = temp
        
        else:
            raise ValueError('Type ID is one of 0,1,2')

        return Chromosome(load_dict, genes)

class Population:
    def __init__(self, chroms):
        self.chroms = chroms
        self.num = len(chroms)

    def display(self):
        for i in self.chroms:
            print(i.genes)
    
    def get_score(self):
        genes = []
        for i in self.chroms:
            genes.append(i.genes)
        input = np.array(genes)
        res = model.inference(IO.compress(input)) * 100
        res = res.reshape((2*self.num,))
        res2 = fitness.var(input)
        res3 = fitness.l2_dis(input)
        # print(res3 * 100000)
        return res + res2 + res3 * 1000
    

    def evolve(self, iter):
        # Reproduction of the population
        sons = []
        for i in self.chroms:
            id = np.random.randint(low=0, high=3)
            sons.append(i.mutate(id))
        self.chroms.extend(sons)
        # get score for each individual
        scores = self.get_score()
        max_id = []
        for i in range(2*self.num):
            max_id.append((i, scores[i]))
        max_id.sort(key=takeSecond, reverse=True)
        select_num = iter // iters * self.num
        new_ids = []
        sons = 0
        parents = 0
        for i in range(self.num):
            # Sons
            if max_id[i][0] >= 32 and sons <= select_num:
                new_ids.append(max_id[i][0])
                sons += 1
                continue
            # Parents
            new_ids.append(max_id[i][0])
            parents += 1
        # print("Next generation:", len(new_ids))

        tmp_set = self.chroms.copy()
        # update pop
        self.chroms.clear()
        for i in new_ids:
            self.chroms.append(tmp_set[i])
    
        

if __name__ == "__main__":
    Net = model.Net

    l = []
    for i in range(num_pop):
        ch = Chromosome(load_dict)
        ch.generate_parent()
        l.append(ch)
    pop = Population(l)
    for i in range(iters):
        pop.evolve(i)
        if i%100 == 0:
            print("Gap: ", i)
            pop.display()

    # ch = Chromosome(load_dict)
    # ch.generate_parent()
    # chan = 0
    # for i in range(iters):
    #     verbose = False
    #     if i % 100 == 0:
    #         verbose = True
    #         print("Search in %s iters"%{i})
    #     id = np.random.randint(low=0, high=3)
    #     son = ch.mutate(id)
    #     ch_g = ch.genes.reshape(1, ch.length)
    #     son_g = son.genes.reshape(1, son.length)
    #     score_ch = model.inference(IO.compress(ch_g)) * 100
        
    #     score_son = model.inference(IO.compress(son_g)) * 100
    #     if score_ch < score_son:
    #         ch = son
    #         chan += 1
    #     if verbose:
    #         print(id)
    #         print(ch.genes)
    #         print(son.genes)
    #         print("score for parent: ", score_ch)
    #         print("score for son: ", score_son)
    #         print("changed: " ,chan)

        
        