# -*- coding: utf-8 -*-
"""
Created on Tue Feb 17 11:15:48 2015

@author: Andreu Mora

Example script to demonstrate the fundamentals of Hidden Markov Models based on
[Rabiner 89] A Tutorial on Hidden Markov Models and Selected Applications in 
Speech Recognition.


"""
import pandas as pd
import numpy as np

class SimpleHMM:
    
    ## Attributes
    # Transition Probability Matrix
    A = [];
    # Symbol emission Probability Matrix
    B = [];
    # State Probability Matrix
    PI = [];
    
    ## Methods
    def __init__(self, aA, aB, aPI):
        self.A = aA;
        self.B = aB;
        self.PI = aPI;
        
    def getOrder (self):
        return len(self.A.index);
        
    def getStatesNames (self):
        return self.A.index;    
        
    def getNumberOfSymbols (self):
        return len(self.B.columns);
        
    def getSymbolsNames (self):
        return self.B.columns;
        
    def computeForwardVariable (self, aSymbolSequence):
        """" Computes the matrix alpha (forward variable) for all the states 
        
        alpha_t (i) = Prob { O1, O2, ..., Ot, q_t = S_i | lambda }    
        """
        # The sequence is given in any form of iterable
        aAlpha = pd.DataFrame(columns=self.getStatesNames(), index=aSymbolSequence);
        # Initialize for each state
        for myState in self.getStatesNames():
            aAlpha[myState].iloc[0] = self.PI[myState]*self.B[myState].loc[aSymbolSequence[0]];
        # Run for all the other emitted symbols
        for i in range(1,len(aSymbolSequence)):
            for myState in self.getStatesNames():
                aAlpha[myState].iloc[i] = ((self.A[myState]*aAlpha.iloc[i-1]).sum())*self.B[myState].loc[aSymbolSequence[i]];
        return aAlpha
        
    def computeBackwardVariable (self, aSymbolSequence):
        """ Computes the matrix beta (backward variable) for all the states
        
        beta_t(i) = Prob { Ot+1, Ot+2, ..., OT, q_t = S_i | lambda }
        """
        aBeta = pd.DataFrame(columns=self.getStatesNames(), index=aSymbolSequence);
        # Initialize the last state to ones per convention
        aBeta.iloc[-1] = np.ones((1,self.getOrder()));
        for i in range(len(aSymbolSequence)-2,-1,-1):
            for myState in self.getStatesNames():
                aBeta[myState].iloc[i] = (aBeta.iloc[i+1]*self.B.loc[aSymbolSequence[i+1]]*self.A.loc[myState]).sum();
        return aBeta;
       
    def computeProbabilityOfSymbolSequence (self, aSymbolSequence):
        """ Computes the probability of a symbol sequence """
        # The probability is the sum at the last 
        myAlpha = self.computeForwardVariable (aSymbolSequence);
        return myAlpha.sum(axis=1).iloc[-1];
        
    def computeViterbi (self, aSymbolSequence):
        """ Computes the Viterbi algorithm for finding the most probable state path """
        
        # Delta equals a transposed Trellis diagram
        aDelta = pd.DataFrame(columns=self.getStatesNames(), index=aSymbolSequence);
        aPath = pd.DataFrame(columns=self.getStatesNames(), index=aSymbolSequence);
        # Initialization
        for myState in self.getStatesNames():
            aDelta[myState].iloc[0] = self.PI[myState]*self.B[myState].loc[aSymbolSequence[0]];
        # Recursive loop
        for i in range(1,len(aSymbolSequence)):            
            for myState in self.getStatesNames():
                aDelta[myState].iloc[i] = self.B[myState].loc[aSymbolSequence[i]]*(self.A.loc[myState]*aDelta.iloc[i-1]).max();
                aPath[myState].iloc[i] = aDelta.columns[np.argmax(self.A.loc[myState]*aDelta.iloc[i-1])];
        return aDelta, aPath;
        
    def computeMostProbableStateSequence (self, aSymbolSequence):
        """ Computes the most probable input sequence for a specific symbol sequence """
        # Compute by Viterbi the best probability and undo the path
        myDelta, myPath = self.computeViterbi(aSymbolSequence);
        # Select the highest probability, if many then choose one
        aProbability = myDelta.iloc[-1].max();
        aStateSequence = pd.Series(index=aSymbolSequence, data=""*len(aSymbolSequence));
        aStateSequence.iloc[-1] = myDelta.columns[np.argmax(myDelta.iloc[-1])];
        for i in range(len(aSymbolSequence)-2,-1,-1):
            # Recall that the path variable is forwarded by 1 unit already
            aStateSequence.iloc[i] = myPath[aStateSequence.iloc[i+1]].iloc[i+1];
        return aStateSequence, aProbability;
        
        
                
        
        
        

# MAIN TEST
myStates = ['H', 'L'];
mySymbols = ['G', 'T', 'C', 'A'];

myPI = pd.Series(data=[0.5, 0.5], index=myStates);

myA = pd.DataFrame(index=myStates, columns=myStates);
myA['H'].loc['H'] = 0.5;
myA['H'].loc['L'] = 0.5;
myA['L'].loc['H'] = 0.4;
myA['L'].loc['L'] = 0.6;

myB = pd.DataFrame(columns=myStates, index=mySymbols);
myB['H'].loc['G'] = 0.3;
myB['H'].loc['T'] = 0.2;
myB['H'].loc['C'] = 0.3;
myB['H'].loc['A'] = 0.2;
myB['L'].loc['G'] = 0.2;
myB['L'].loc['T'] = 0.3;
myB['L'].loc['C'] = 0.2;
myB['L'].loc['A'] = 0.3;

myHMM = SimpleHMM(myA, myB, myPI);

mySymbolSequence = ['G', 'G', 'C', 'A', 'C', 'T', 'G', 'A', 'A']
myStateSequence, myProb = myHMM.computeMostProbableStateSequence(mySymbolSequence);
print myStateSequence
print myProb;
    
    