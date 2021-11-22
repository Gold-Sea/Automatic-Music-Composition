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
        self.length = jsonstr['bars'] * jsonstr['pluses']
        self.num_tone = num_tone + 2
        self.geneset = [i for i in range(self.num_tone)]
        self.genes = None
    
    # return a numpy array with shape(num_tones + 2,)
    def generate_parent(self): 
        return np.random.randint(low=0, high=self.num_tone,size=(self.length,))

    def mutate(self, typeid):
        if typeid == 0:
            randindex = np.random.randint(low=0, high=self.length)
            if_pluse = np.random.randint(low=0, high=2)
            original_val = self.genes[randindex]
            # Upward overflow
            if original_val + 8 >= self.num_tone:
                if_pluse = 0
            if original_val - 8 < 0:
                if_pluse = 1
            if if_pluse:
                self.genes[randindex] += 8
            else:
                self.genes[randindex] -= 8

        elif typeid == 1:
            randindex = np.random.randint(low=0, high=self.length)
            original_val = self.genes[randindex]
            randval = np.random.randint(low=0, high=self.length)
            while(randval != original_val):
                randval = np.random.randint(low=0, high=self.length)
            self.genes[randindex] = randval

        elif typeid == 2:
            randindex = np.random.randint(low=0, high=self.length - 1)
            temp = self.genes[randindex]
            self.genes[randindex] = self.genes[randindex + 1]
            self.genes[randindex + 1] = temp
        
        else:
            raise ValueError('Type ID is one of 0,1,2')

    
        

if __name__ == "__main__":
    pass