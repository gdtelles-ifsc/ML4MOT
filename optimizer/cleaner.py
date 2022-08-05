import numpy as np
import os
import csv

def timescale():
    # timescale for when the x is in points and not seconds
    return np.float64(15.03/1320)


def formatting(data_path):
    # loading for txt files only
    curve = open(data_path,'r')
    firstline,*curvelines = curve.readlines()
    curve.close()
    os.remove(data_path)

    loading_curve = np.empty([0,2])
        
    for line in curvelines:
        formatted_line = np.array([np.float64([line.split('\t')[0],line.split('\t')[1][:-2]])])
        loading_curve = np.concatenate((loading_curve,formatted_line*[timescale(),1]))
    loading_curve -= np.array([loading_curve[0]]*len(loading_curve))
        
        #Later, t0 should be informed in a specific way in the first line. 
        #The formatting function will then be able to return it alongside
        #the curve points
        
    t0 = np.float64(firstline[-5:-2])
       
    return t0,loading_curve


def load_curve(data_path):
    # loading for csv files
    
    curve = np.empty([0,2])
    with open(data_path, newline='') as file:
        curve_reader = csv.reader(file, delimiter=',', quotechar='|')
        i = 1
        for row in curve_reader:
            if i:
                i = 0
            else:
                curve = np.concatenate((curve, [[np.float64(row[0]), np.float64(row[1])]]),axis=0)
                
    return curve


def reset_t0(t0,curve):
    # must be called before reset_n0
    curve = np.array([point for point in curve if point[0]>=t0])     
    curve = np.array([point - [curve[0, 0], 0] for point in curve])
    return curve


def reset_n0(curve):
    # assumes reset_t0 has already been called
    curve = np.array([point - curve[0] for point in curve])
    curve = np.array([point for point in curve if point[1]>=0])  
    return curve    

def reset_origin(t0, curve):
    # use this to make sure the resets are called in the right order
    curve = reset_t0(t0, curve)
    curve = reset_n0(curve)
    return curve