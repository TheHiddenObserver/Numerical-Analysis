# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 20:54:28 2020

@author: linziqian
"""

import numpy as np

class Spline:
    def __init__(self, x, y):
        self.__x = x
        self.__y = y
        self.__m = None
        self.__type = ""
        self.__lamb = None
        self.__mu = None
    
    
    @property
    def get_curr_type(self):
        return self.__type
    
    @property
    def get_lambda(self):
        return self.__lamb
    
    @property
    def get_mu(self):
        return self.__mu
    
    @property
    def get_m(self):
        return self.__m
    
    
    def __calculate_diff(self, add_condition, method = "D1"):
        '''
        Calculate the first order deravitives of the given points.
        '''
        x, y = self.__x, self.__y
        n, h = len(x), np.diff(x)
        y_diff = np.diff(y)
        lamb = [0] * (n - 2)
        mu = [0] * (n - 2)
        for i in range(1,n-1):
            lamb[i-1] = h[i-1] / (h[i-1] + h[i])
            mu[i-1] = 3 * ((1 - lamb[i-1]) / h[i-1] * (y_diff[i-1]) + lamb[i-1] / h[i] * y_diff[i])
            
        self.__lamb = lamb
        
        if method == "D1":
            if len(add_condition) != 2:
                raise ValueError("Additional Condition Length Error")
            
            mu[0] = mu[0] - (1 - lamb[0]) * add_condition[0]
            mu[-1] = mu[-1] - lamb[-1] * add_condition[1]
            mu = np.mat(mu).T
            self.__mu = mu
            
            # construct the matrix
            matrix = [[0] * (n - 2) for _ in range(n - 2)]
            for i in range(n-2):
                matrix[i][i] = 2
                if i > 0:
                    matrix[i-1][i] = lamb[i-1]
                if i < n - 3:
                    matrix[i+1][i] = 1 - lamb[i-1]
            matrix = np.mat(matrix)
            
            
            self.__m = [add_condition[0]] + list(np.asarray(np.linalg.solve(matrix, mu)).reshape(n-2)) + [add_condition[1]]
            self.__type = "D1"
            
        elif method == "D2":
            if len(add_condition) != 2:
                raise ValueError("Additional Condition Length Error")
                
            mu0 = 3 * y_diff[0] / h[0] - h[0] / 2 * add_condition[0]
            mun = 3 * y_diff[-1] / h[-1] + h[-1] / 2 * add_condition[1]
            mu.insert(0, mu0)
            mu.append(mun)
            mu = np.mat(mu).T
            self.__mu = mu
            
            # construct the matrix
            matrix = [[0] * n for _ in range(n)]
            matrix[0][0], matrix[0][1], matrix[-1][-2], matrix[-1][-1] = 2,1,1,2
            for i in range(1,n-1):
                matrix[i][i-1], matrix[i][i], matrix[i][i+1] = 1 - lamb[i-1], 2, lamb[i-1]
            matrix = np.mat(matrix)
            
            self.__m = list(np.asarray(np.linalg.solve(matrix, mu).T).reshape(n))
            self.__type = "D2"
            
            
    def __calculate_value(self, new_points):
        '''
        Calculate the interpolated value for the new points.
        '''
        x, y, m = self.__x, self.__y, self.__m
        index = 0
        n = len(x)
        if new_points[-1] > x[-1] or new_points[0] < x[0]:
            raise ValueError("New points are out of range.")
        res = []
        h = np.diff(x)
        
        for point in new_points:
            # find the interval the point belonging to 
            while index < n and x[index] <=  point:
                index += 1
            index -= 1
            if index == n - 1:
                index -= 1
                
            # Use the Hermite interpolation polynomial to calculate.
            tmp = (1 + 2 * (point - x[index]) / h[index]) * ((point - x[index+1]) / h[index]) ** 2 * y[index] + \
                (1 - 2 * (point - x[index+1]) / h[index]) * ((point - x[index]) / h[index]) ** 2 * y[index+1] + \
                (point - x[index]) * ((point - x[index+1]) / h[index]) ** 2 * m[index] + \
                (point - x[index+1]) * ((point - x[index]) / h[index]) ** 2 * m[index+1]
            res.append(tmp)
            
        return res
    
    
    def spline(self, new_points, add_condition, method = "D1"):
        '''
        Calculate the cubic spline interpolation. 
        Notice: The parameter new_points should be sorted.
        '''
        x, y = self.__x, self.__y
        if len(x) != len(y):
            raise ValueError("The lengths of x and y are not matched.")
        x, y = zip(*sorted(zip(x,y)))
        self.__x, self.__y = np.array(x), np.array(y)
        
        self.__calculate_diff(add_condition, method)
        res = self.__calculate_value(new_points)
        return res
    
    
if __name__ == '__main__':
    x = np.arange(0,3.1,0.6)
    y = np.exp(- x / 4)
    inter = Spline(x,y)
    
    diff = [-1/4,-1/4 * np.exp(-3/4)] #first order conditions
    diff2 = [1/16, 1/16 * np.exp(-3/4)] # second order conditions
    r = np.linspace(0,3,7)
    y_true = np.exp(-r / 4)
    
    y_inter = inter.spline(r, diff, method = "D1")
    y_inter2 = inter.spline(r, diff2, method = "D2")
    print(y_true - y_inter, y_true - y_inter2, sep = '\n')