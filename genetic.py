import json
import numpy as np

with open("./GA.json",'r') as load_f:
    load_dict = json.load(load_f)
    num_tones = load_dict['num_tones']


class Chromosome:
    def __init__(self, jsonstr, fitness):
        # fitness is a func
        assert isinstance(fitness, function)
        assert isinstance(jsonstr, dict)

        self.Fitness = fitness
        num_tone = jsonstr['num_tones']
        self.length = num_tone + 2
        self.geneset = [i for i in range(self.length)]
        self.genes = None
    
    # return a numpy array with shape(num_tones + 2,)
    def generate_parent(self): 
        return np.random.randint(low=0, high=self.length,size=(self.length,))

    def mutate(self, typeid):
        pass

    
        

if __name__ == "__main__":
    print(np.random.randint(low=0, high=5,size=(5,)))