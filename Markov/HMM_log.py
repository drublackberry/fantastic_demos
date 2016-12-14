# -*- coding: utf-8 -*-
"""
Created on Tue Feb 17 11:15:48 2015

@author: Andreu Mora
"""
import pandas as pd
import numpy as np
import scipy.stats

class rv_string ():
    strIndex = [];
    rvg = [];

    def __init__(self, aString,aProb):
        # Index the string as integer
        self.strIndex = pd.Series(aString)
        self.rvg = scipy.stats.rv_discrete(values=(self.strIndex.index, aProb))
        
    def generate (self, aN):
        # Generates N characters
        aGenIndex = self.rvg.rvs(size=aN);
        # Translate the indexes to characters
        return np.array(self.strIndex.loc[aGenIndex])

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
        return self.B.index;
        
    def simulateSequence (self, aN):
        """"
        Simulates a state and symbol sequence based on the model
        """
        aQ = [];
        aO = [];
        # Create all the random generators
        myRVGstate = {};
        myRVGsymbol = {}
        myRVGstate['init'] = rv_string(self.getStatesNames(), np.array(self.PI));
        # For each state
        for myState in self.getStatesNames():
            myRVGstate[myState] = rv_string(self.A.columns,np.array(self.A[myState]));
            myRVGsymbol[myState] = rv_string(self.B.index, np.array(self.B[myState]));
            
        # Initialization
        aQ.append(myRVGstate['init'].generate(1)[0]);
        aO.append(myRVGsymbol[aQ[0]].generate(1)[0]);
        # State propagation and symbol creation
        for i in range(1,aN):
            aQ.append(myRVGstate[aQ[i-1]].generate(1)[0]);
            aO.append(myRVGsymbol[aQ[i]].generate(1)[0]);
        return aQ, aO;
            

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
        # Translate the probabilities to the log domain
        myLogA = self.A.astype('float').apply(np.log2);
        myLogB = self.B.astype('float').apply(np.log2);
        myLogPI = self.PI.astype('float').apply(np.log2);
        
        # Delta equals a transposed Trellis diagram
        aDelta = pd.DataFrame(columns=self.getStatesNames(), index=aSymbolSequence);
        aPath = pd.DataFrame(columns=self.getStatesNames(), index=aSymbolSequence);
        # Initialization
        for myState in self.getStatesNames():
            aDelta[myState].iloc[0] = myLogPI[myState]+myLogB[myState].loc[aSymbolSequence[0]];
        # Recursive loop
        for i in range(1,len(aSymbolSequence)):            
            for myState in self.getStatesNames():
                aDelta[myState].iloc[i] = myLogB[myState].loc[aSymbolSequence[i]]+(myLogA.loc[myState]+aDelta.iloc[i-1]).max();
                aPath[myState].iloc[i] = aDelta.columns[np.argmax(myLogA.loc[myState]+aDelta.iloc[i-1])];
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
myStates = ['S1', 'S2', 'S3', 'S4'];
mySymbols = ['goodalpha', 'midalpha', 'badalpha'];

myPI = pd.Series(data=[0.5, 0.5, 0.5, 0.5], index=myStates);

myA = pd.DataFrame(index=myStates, columns=myStates);
myA['S1'].loc['S1'] = 0.5;
myA['S1'].loc['S2'] = 0.5;

myA['S2'].loc['S1'] = 1/3.;
myA['S2'].loc['S2'] = 1/3.;
myA['S2'].loc['S3'] = 1/3.;

myA['S3'].loc['S2'] = 1/3.;
myA['S3'].loc['S3'] = 1/3.;
myA['S3'].loc['S4'] = 1/3.;

myA['S4'].loc['S3'] = 1/2.;
myA['S4'].loc['S4'] = 1/2.;


myB = pd.DataFrame(columns=myStates, index=mySymbols);
myB['S1'].loc['goodalpha'] = 1/6.;
myB['S1'].loc['midalpha'] = 1/6.;
myB['S1'].loc[''] = 1/6.;


myN = 100;

myHMM = SimpleHMM(myA, myB, myPI);
myQ, myO = myHMM.simulateSequence(myN);

#mySymbolSequence = ['G', 'G', 'C', 'A', 'C', 'T', 'G', 'A', 'A']
mySymbolSequence = myO;
myStateSequence, myProb = myHMM.computeMostProbableStateSequence(mySymbolSequence);

# check the states
myCheckDF = pd.DataFrame(index=myStateSequence.index, columns=['True', 'Estimated', 'Equal']);
myCheckDF['True'] = myQ;
myCheckDF['Estimated'] = myStateSequence;
myCheckDF['Equal'] = myCheckDF['True'] == myCheckDF['Estimated'];
print 'Matched ' + str(100.*myCheckDF['Equal'].sum()/myN) + '%'

    
    