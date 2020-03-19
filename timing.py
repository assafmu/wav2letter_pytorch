# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 10:39:23 2020

@author: User
"""
import time
import random
from collections import defaultdict

def _mock_log(d):
    print(d)
    
    
def _mock_tensorboard_log(epoch,values):
    values.update({'Epoch':epoch})
    print(values)
    
def _slow_generator(i,t=0.1):
    for j in range(i):
        time.sleep(t)
        yield j

def _slow_action(t,res=1):
    time.sleep(t)
    return res
        
def time_generator_average(generator,callback=_mock_log):
    total_time = 0
    start_time = time.time()
    n=0
    for i in generator:
        total_time += time.time() - start_time
        start_time = time.time()
        n+=1
        yield i
    callback(total_time / n)


def average_across_epoch(epoch,name,generator,callback=_mock_tensorboard_log):
    total_time = 0
    n=0
    start_time = time.time()
    for i in generator:
        total_time += time.time() - start_time
        n+=1
        yield i
        start_time = time.time()
    callback(epoch=epoch,values={name:total_time / n})
        

class EpochTimer(object):
    def __init__(self,epoch,logging_callback=_mock_tensorboard_log):
        self.metrics = defaultdict(float)
        self.counts = defaultdict(int)
        self.epoch = epoch
        self.callback = logging_callback
    def __enter__(self):
        return self
    def __exit__(self,type,value,traceback):
        self.callback(self.epoch,{k: self.metrics[k]/self.counts[k] for k in self.metrics})
    def timed_action(self,name):
        return EpochTimedAction(self,name)
    def add(self,name,value):
        self.metrics[name]+=value
        self.counts[name]+=1
    def across_epoch(self,name,generator):
        return average_across_epoch(self.epoch,name,generator,self.callback)
           
class EpochTimedAction(object):
    def __init__(self,parent,name):
        self.start_time = None
        self.parent = parent
        self.name = name
    def __enter__(self):
        self.start_time = time.time()
    def __exit__(self,type,value,traceback):
        self.parent.add(self.name,time.time() - self.start_time)


def _example_usage():
    epochs=3
    for epoch in range(epochs):
        with EpochTimer(epoch) as et:
            for data in et.across_epoch('Data Loading',_slow_generator(4,t=0.01)):
                prepared_data = True # Or other actions
                with et.timed_action('Model execution'):
                    bla = _slow_action(0.2)
                bla = bla + 1 #We  have access to bla
                with et.timed_action('Back propagation'):
                    _slow_action(bla / 5)
                with et.timed_action('Model evaluation'):
                    _slow_action(0.1)
                
                

if __name__ == '__main__':
    _example_usage()