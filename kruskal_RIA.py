'''
Kruskal: relative importance analysis
'''

''' load modules '''
import numpy as np
import itertools
from math import factorial
from numpy import matrix, linalg

''' function to calculate R2 for a subset of the matrix '''
def calcR2(varlist, rmatrix):
    #only include variables in the list
    workmat = rmatrix[np.ix_(varlist, varlist)]    
    #take inverse of workmat
    workmat = linalg.inv(workmat)
    rcx = rmatrix[np.ix_(varlist),0]
    rcxh = rcx.T
    weights = rcx * workmat
    r2 = weights * rcxh
    return r2[0,0]

''' Kruskal calculation '''

def Kruskal(correlationmatrix):
    '''
    This function calculates relative importance of a range of independent variables on
    a dependent variable.
    
    Input: correlation matrix
    
    make sure that the dependent variable comes first in your correlation matrix
    '''
    #assert that there are no negative correlations
    assert all([0 <= el <= 1 for el in enumerate(correlationmatrix)])
    
    #turn input into matrix structure
    correlationmatrix = matrix(correlationmatrix)
    
    noVars = len(correlationmatrix) #number of independent variables
    
    #calculate the factors to multiply the (semipartial) correlations with (in diagonal)
    T = [[factorial(n) * factorial(k) for n in range(noVars - 1)] for k in range(noVars - 1)][::-1]
    
    #create structures to save output
    mean_semipart = np.zeros((noVars - 1))
    list_semipart = []
    
    ##start the calculation
    
    #loop over each independent variable (IV)
    for IV in range(1, noVars):
        
        #make a list of the remaining control variables
        CV_list = list(range(1, noVars))
        CV_list.remove(IV)
        
        #if no control variables => take squared correlation
        mean_semipart[IV - 1] = (correlationmatrix[0, IV] ** 2) * T[0][0]
        list_semipart.append([correlationmatrix[0, IV] ** 2])
        
        #loop over all possible combinations of control variables (CV)
        for CV in [x for l in range(1, noVars + 1) for x in itertools.combinations(CV_list, l)]:
            
            #calculate R2 full (= R2 with the independent variable and the control variables)
            full_list = list(CV)
            full_list.append(IV)
            
            R2full = calcR2(full_list, correlationmatrix)
            #calculate R2 cont (= R2 with only the control variables)
            R2cont = calcR2(list(CV), correlationmatrix)
            #calculate semipartial correlation
            semipart = R2full - R2cont
            #store the semipart as a list for each IV
            list_semipart[IV - 1].append(semipart)
            
            #add to sum of mean squared semipartial correlation    
            #weight => see https://oeis.org/A098361
            mean_semipart[IV - 1] += semipart * T[len(CV)][len(CV)]
      
        mean_semipart[IV - 1] /= factorial(noVars - 1)
    
    
    #print the output
    print("\tRelative importance (%)\tMean of squared semipartial correlations\tMin\tMax")
    template = "Variable {}\t{}\t{}\t{}\t{}\n"
    for IV in range(noVars - 1):
        print(template.format(IV + 1, mean_semipart[IV]/sum(mean_semipart), mean_semipart[IV], min(list_semipart[IV]), max(list_semipart[IV])))

