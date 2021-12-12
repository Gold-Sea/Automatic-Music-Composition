import json
import numpy as np
from numpy.lib.function_base import append, select
import model
import IO
import fitness
import random
import sound

def takeSecond(elem):
    return elem[1]

def swap(l, i1, i2):
    temp = l[i1]
    l[i1] = l[i2]
    l[i2] = temp

pro = [90, 3, 3, 3, 1]
p = []
for i in range(len(pro)):
    if i == 0:
        p.append(pro[i])
    else:
        p.append(pro[i] + p[i - 1])

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

    # 变异操作
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

    # 交叉操作
    def crossover(self, other_genes):
        p1 = self.genes.copy()
        p2 = other_genes.copy()
        length = len(p1)
        assert length==len(p2)
        res = [0 for i in range(length)]
        index1 = random.sample(range(0, length), length//2)
        index2 = []
        for i in range(length):
            if i not in index1:
                index2.append(i)
        for i in index1:
            res[i] = p1[i]
        for i in index2:
            res[i] = p2[i]
        res = np.array(res)
        return Chromosome(load_dict, res)

    # 移调操作 
    def transpotion(self):
        genes = self.genes.copy()
        upper = self.num_tone - 1
        lower = 0
        max_v = -1
        min_v = 1e9
        for i in genes:
            if i !=upper and i!=lower:
                max_v = max(max_v, i)
                min_v = min(min_v, i)
        lower = lower - min_v + 1
        upper = upper - max_v - 1
        if lower == upper:
            randv = 0
        else:
            randv = np.random.randint(low=lower, high=upper)
        res = []
        for i in genes:
            if i == 0 or i == self.num_tone - 1:
                res.append(i)
            else:
                res.append(randv + i)
        res = np.array(res)
        return Chromosome(load_dict, res)


    #逆行变换
    def retrograde(self):
        genes = self.genes.copy()
        # traverse list to sink all extended notes
        extend = self.num_tone - 1
        for i in range(1, len(genes)):
            if genes[i] == extend:
                swap(genes, i, i-1)
        
        genes = genes.tolist()
        genes.reverse()
        genes = np.array(genes)
        return Chromosome(load_dict, genes)

    #倒影变换
    def inversion(self):
        # Criteria for selecting mirror images
        cen_note = self.num_tone // 2
        genes = []
        for i in self.genes:
            if i == 0 or i == self.num_tone - 1:
                genes.append(i)
                continue
            delta = i - cen_note
            tmp = cen_note - delta
            # Prevent exceeding the border
            if tmp < 1:
                tmp = 1
            if tmp > self.num_tone - 2:
                tmp = self.num_tone - 2
            genes.append(tmp)
        genes = np.array(genes)
        return Chromosome(load_dict, genes)
    
    def playsound(self):
        sound.play_sequence(list(self.genes))
        


class Population:
    def __init__(self, chroms):
        self.chroms = chroms
        self.num = len(chroms)

    def display(self, sound = False):
        cnt = 0
        print(self.get_score())
        for i in self.chroms:
            print(i.genes)
            if (sound and cnt < 1): # play first 3 chroms
                i.playsound()
                cnt += 1
    
    def get_score(self):
        genes = []
        for i in self.chroms:
            genes.append(i.genes)
        input = np.array(genes)

        res = model.inference(IO.compress(input)) * 100
        #print(len(self.chroms))
        #print(len(input))
        #print(len(res))
        res = res.reshape((len(self.chroms),))
        res2 = fitness.var(input)
        #res3 = fitness.l2_dis(input)
        # print(res3 * 100000)
        #return res + res2 + res3 * 1000
        # try different combinations of fitness functions
        return 0*fitness.l2_dis(input) + 1*fitness.knn(input)


    def evolve(self, iter):
        # Reproduction of the population
        sons = []
        '''The probability of selecting five methods of generating 
        offspring: mutate, crossover, transpotion, retrograde, inversion'''
        global p
        probability = np.random.randint(low=1, high=101)
        length = len(self.chroms)
        for ik in range(length):
            if probability <= p[0]:
                id = np.random.randint(low=0, high=3)
                index = np.random.randint(low=0, high=length)
                sons.append(self.chroms[index].mutate(id))
            elif probability <= p[1]:
                parents = random.sample(range(0, length), 2)
                sons.append(self.chroms[parents[0]].crossover(self.chroms[parents[1]].genes))
            elif probability <= p[2]:
                index = np.random.randint(low=0, high=length)
                sons.append(self.chroms[index].transpotion())
            elif probability <= p[3]:
                index = np.random.randint(low=0, high=length)
                sons.append(self.chroms[index].retrograde())
            elif probability <= p[4]:
                index = np.random.randint(low=0, high=length)
                sons.append(self.chroms[index].inversion())
        # for i in self.chroms:
        #     id = np.random.randint(low=0, high=3)
        #     sons.append(i.mutate(id))
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
        if i%50 == 0:
            print("Gap: ", i)
            pop.display(sound=True)

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

        
        