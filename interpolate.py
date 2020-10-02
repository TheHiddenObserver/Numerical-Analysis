# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 17:39:32 2020

@author: linziqian
"""

import numpy as np

class Interpolate:
    def __init__(self,x = None,y = None):
        '''
        x: 被插节点
        y: 插值条件
        '''
        self.__x = np.array(x)
        self.__y = np.array(y)
        self.diff = [[] for _ in range(len(x))] if x is not None else []
        
    def update_points(self, x, y):
        self.__x = np.array(x)
        self.__y = np.array(y)
        self.diff = [[] for _ in range(len(x))]
        self.calculate_diff()
    
    def Lagrange(self, r):
        '''
        Lagrange Interpolation.
        Return an array of interpolation result.
        r: the new points
        '''
        x, y = self.__x, self.__y
        if len(x) != len(y):
            raise ValueError("Length Error")
        res = []
        for point in r:
            tmp = []
            for i in range(len(x)):
                pro = 1
                for j in range(len(y)):
                    if i != j:
                        pro *= (point - x[j]) / (x[i] - x[j])
                tmp.append(pro)
            res.append(np.dot(y,np.array(tmp)))
        return res
    
    def Linear(self,r):
        '''
        Successive linear interpolation method.
        '''
        x, y = self.__x, self.__y
        if len(x) != len(y):
            raise ValueError("Lengths are not equal")
        n = len(x)
        p = [[0] * n for _ in range(n)]
        for i in range(n):
            p[i][i] = y[i]
        res = []
        for point in r:
            for k in range(n-1):
                for i in range(n - k - 1):
                    p[i][i+k+1] = p[i][i+k] + (p[i+1][i+k+1] - p[i][i+k]) / (x[i+k+1] - x[i]) * (point - x[i])
            res.append(p[0][n-1])
        return res
    
    def calculate_diff(self):
        '''
        Calculate the difference quotient for the interpolators.
        '''
        x, y = self.__x, self.__y
        n = len(x)
        self.diff[0] = list(y)
        for k in range(1,n):
            for i in range(n - k):
                self.diff[k].append((self.diff[k-1][i+1] - self.diff[k-1][i]) / (x[i+k] - x[i]))
        
    def Newton(self,r):
        '''
        Newton method.
        '''
        x, y = self.__x, self.__y
        if len(x) != len(y):
            raise ValueError("Lengths are not equal")
        n = len(x)
        if not self.diff[0]:
            self.calculate_diff()
        diff_new = np.array([self.diff[i][0] for i in range(n)])
        res = []
        for point in r:
            val = [1]
            for i in range(n-1):
                val.append(val[-1] * (point - x[i]))
            val = np.array(val)
            res.append(np.dot(diff_new, val))
        return res
    
    def calculate_error(self,r,y_e,fun):
        '''
        Calculate the calculation error for interpolation.
        '''
        y_t = fun(r)
        return y_t - y_e
    
if __name__ == '__main__':
    r = np.linspace(0,1,11)
    fun = lambda x: np.exp(-x/4)
    
    x = np.linspace(0,1,3)
    y = np.exp(-x/4)
    inter = Interpolate(x,y)
    print(inter.calculate_error(r,inter.Lagrange(r), fun),
          inter.calculate_error(r,inter.Linear(r), fun),
          inter.calculate_error(r,inter.Newton(r), fun),sep = '\n')
    
    x_new = np.linspace(0,1,4)
    y_new = np.exp(-x_new/4)
    inter_new = Interpolate(x_new,y_new)
    print(inter_new.calculate_error(r,inter_new.Lagrange(r), fun),
          inter_new.calculate_error(r,inter_new.Linear(r), fun),
          inter_new.calculate_error(r,inter_new.Newton(r), fun),sep = '\n')
    
    inter2 = Interpolate()
    inter2.update_points(x, y)
    print(inter2.calculate_error(r,inter.Lagrange(r), fun),
          inter2.calculate_error(r,inter.Linear(r), fun),
          inter2.calculate_error(r,inter.Newton(r), fun),sep = '\n')