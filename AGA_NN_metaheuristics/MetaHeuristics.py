import intelligence

import numpy as np
import random
import time
###python packages
from math import ceil, exp
from random import choice, shuffle
import warnings
import pandas as pd
from random import randint, uniform
from tqdm import tqdm

def no_dup(L_0): # remueve valores duplicados de lista
    L_n = []
    L = []
    index = 0
    for x in L_0:
        if x not in L_n:
            L.append(index)
            L_n.append(L_0[index])
        index += 1
    return  L

           
class aga(intelligence.sw):
    """
    Asexual Genetic Algorithm
    """
    def __init__(self, n, nk,  function, lb, ub, dimension, iteration):
        """
        :param n: number of agents
        :param nk: number of agents to keep
        :param function: test function
        :param lb: lower limits for plot axes
        :param ub: upper limits for plot axes
        :param dimension: space dimension
        :param iteration: number of iterations
        """

        super(aga, self).__init__()
        self.__function = function
        self.__ulim = ub
        self.__llim = lb
        self.__std_dev = self.__ulim - self.__llim
        self.__agents  =  np.random.uniform(self.__llim,self.__ulim, (n,dimension))
        self.__agents_fit = np.array([function(x) for x in self.__agents])
        Pbest = self.__agents[np.array([function(x)
                                        for x in self.__agents]).argmin()]
        fitness = [function(x) for x in self.__agents]
        Gbest = Pbest.copy()
        z = []
        Gb = []
        Gb.append(Gbest)
        z.append(function(Gbest))
        t_0 = time.time()
        for t in tqdm(range(iteration)):
            self.__agents_fit = np.array([function(x) for x in self.__agents])
            self._points(self.__agents)            
            topIdx = np.argsort(self.__agents_fit)
            self.__parents_top  = self.__agents[ topIdx[0:nk] , : ]
            self.__parents_top_fit = np.array([function(x) for x in self.__parents_top])
            self.__std_dev = np.array([np.std(x) for x in self.__parents_top.T])
            for i in range(0,nk):
                 self.__agents[nk*i] = self.__parents_top[i].copy()
                 self.__ulim = np.minimum(self.__parents_top[i] + self.__std_dev, ub)
                 self.__llim = np.maximum(self.__parents_top[i] - self.__std_dev, lb)   
                 for ink in range(1,nk):
                     for j in range(dimension):
                        self.__agents[nk*i+ink][j] = np.random.uniform(self.__llim[j],self.__ulim[j] )
            Pbest = self.__agents[
                 np.array([function(x) for x in self.__agents]).argmin()]
            if np.abs(function(Pbest)) < np.abs(function(Gbest)):
                Gbest = Pbest.copy()
            Gb.append(Gbest)
            z.append(function(Gbest))
        t_f =  time.time()  - t_0         
        self._set_time(t_f)
        self._set_Gbest(Gbest)
        self._set_zhistory(z)
        self._set_history(Gb)
        
        
        

class de(intelligence.sw):
    """
    Diferential Evolution
    """

    def __init__(self, n, function, lb, ub, dimension, iteration, F, cr):
        """
        :param n: number of agents
        :param function: test function
        :param lb: lower limits for plot axes
        :param ub: upper limits for plot axes
        :param dimension: space dimension
        :param iteration: the number of iterations
        :param w: balance between the range of research and consideration for
        suboptimal decisions found (default value is 0.5):
        w>1 the particle velocity increases, they fly apart and inspect
         the space more carefully;
        w<1 particle velocity decreases, convergence speed depends
        on parameters c1 and c2 ;
        :param c1: ratio between "cognitive" and "social" component
        (default value is 1)
        :param c2: ratio between "cognitive" and "social" component
        (default value is 1)
        """

        super(de, self).__init__()
        self.__agents = np.random.uniform(lb,ub, (n,dimension))
        self.__ulim = ub
        self.__llim = lb

        self._points(self.__agents)
        Pbest = self.__agents[np.array([function(x)
                                        for x in self.__agents]).argmin()]
        Gbest = Pbest.copy()
        z = []
        Gb = []
        Gb.append(Gbest)
        z.append(function(Gbest))
        fitness = [function(x) for x in self.__agents]
        t_0 = time.time()
        for t in tqdm(range(iteration)):
            for i in np.arange(n):
                a,b,c = np.random.randint((n,n,n))
                mutated = self.__mutation([self.__agents[a],self.__agents[b],self.__agents[c]],F) 
                mutation = self.__check_bounds( mutated,dimension,n,F)
                trial = self.__crossover(mutated, self.__agents[i], dimension, cr)
                trial = self.__check_boundsx( trial,self.__agents[i],dimension,n,cr)
                obj_trial = function(trial)
                if obj_trial < fitness[i]:
                    self.__agents[i] = trial
                    fitness[i] = obj_trial
            Gbest = self.__agents[
                np.array([function(x) for x in self.__agents]).argmin()]
            Gb.append(Gbest)
            z.append(function(Gbest))
        t_f =  time.time()  - t_0         
        self._set_time(t_f)
        self._set_Gbest(Gbest)
        self._set_zhistory(z)
        self._set_history(Gb)
    def __mutation(self, x,F):
        return x[0] + F*(x[1]-x[2])
    def __check_bounds(self, x,dimension,n,F):
        for j in range(dimension):
           while (x[j] < self.__llim[j] or x[j]>self.__ulim[j]):
               a,b,c = np.random.randint((n,n,n))
               x = self.__mutation([self.__agents[a],self.__agents[b],self.__agents[c]],F) 
        return x
    def __check_boundsx(self, x,y,dimension,n,cr):
        for j in range(dimension):
            while (x[j] < self.__llim[j] or x[j]>self.__ulim[j]):
               a,b,c = np.random.randint((n,n,n))
               x= self.__crossover(x, y, dimension, cr) 
        return x
    def __crossover(self,mutated, target, dimension, cr):
        # generate a uniform random value for every dimension
        p = np.random.random(dimension)
        # generate trial vector by binomial crossover
        trial = [mutated[i] if p[i] < cr else target[i] for i in range(dimension)]
        return trial
