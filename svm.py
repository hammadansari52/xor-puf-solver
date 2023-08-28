from math import sqrt
import numpy as np
import time as t
class moiSVM():
    def __init__(self, C, init_lr, X, y, lr_mode):
        self.C = C
        self.init_lr = init_lr
        self.X = X #X is n,d shape
        self.y = y
        self.w = np.ones(X.shape[1])
        self.lr_mode = lr_mode

    def doStochasticMBGD(self, n, lr):
        # w is d+1, X is n*d+1
        sampled_ind = np.random.choice(self.X.shape[0], size=n, replace=True)
        yx = self.y[sampled_ind][:,None]*self.X[sampled_ind] # n, 3
        g = np.where(yx.dot(self.w)>1, 0, -1) # n
        delw = self.w + self.C*(yx.T).dot(g)
        self.w = self.w - lr*delw

    def primal_obj(self):
        return 0.5*(np.linalg.norm(self.w)**2) + self.C*np.sum(np.maximum(0, 1-self.y*self.w.dot(self.X.T)))

    def train_SVM(self, n, n_steps):
        train_loss = []
        for i in range(n_steps):
            if self.lr_mode=='exp':
                lr = self.init_lr*(0.99999**i)
            if self.lr_mode=='sqrt':
                lr = self.init_lr/sqrt(i+1)
            self.doStochasticMBGD(n, lr)
            if i%10==0:
                train_loss.append(self.primal_obj())
        return train_loss
        
def getRandpermCoord( state ):
    idx = state[0]
    perm = state[1]
    d = len( perm )
    if idx >= d - 1 or idx < 0:
        idx = 0
        perm = np.random.permutation( d )
    else:
        idx += 1
    state = (idx, perm)
    curr = perm[idx]
    return (curr, state)

# Get functions that offer various coordinate selection schemes
def coordinateGenerator( mode, d ):
    if mode == "cyclic":
        return (getCyclicCoord, (0,d))
    elif mode == "random":
        return (getRandCoord, d)
    elif mode == "randperm":
        return (getRandpermCoord, (0,np.random.permutation( d )))

class softsirSVM():
    def __init__(self, C, X, y):
        self.C = 1

        self.coordFunc = coordinateGenerator( "randperm", y.size )
        
        self.X = X #should be n*729
        print('X shape is' , X.shape)
        self.alpha = C * np.ones( ( y.size, ) )
        self.y = y
        self.normSq = np.square( np.linalg.norm( X, axis = 1 ) )+1

        self.w_SDCM = X.T.dot(np.multiply( self.alpha, y )) #729
        print('weights shape is',self.w_SDCM.shape)

        self.b_SDCM = (self.alpha).dot( y )

    def mySVM(self, X ):
        return X.dot(self.w_SDCM) + self.b_SDCM

    # Stochastic Dual Coordinate Maximization
    def doCoordOptCSVMDual(self, i):
    
        x = self.X[i,:]
        
        # Find the unconstrained new optimal value of alpha_i
        # It takes only O(d) time to do so because of our clever book keeping
        p = self.y[i] * (x.dot(self.w_SDCM) + self.b_SDCM - self.alpha[i]*self.y[i]*self.normSq[i])
        # p = self.y[i] * (x.dot(self.w_SDCM) - self.alpha[i]*self.y[i]*self.normSq[i])
        newAlphai =  (1 - p) / self.normSq[i]
        
        # Make sure that the constraints are satisfied. This takes only O(1) time
        if newAlphai > self.C:
            newAlphai = self.C
        if newAlphai < 0:
            newAlphai = 0

        # Update the primal model vector and bias values to ensure bookkeeping is proper
        # Doing these bookkeeping updates also takes only O(d) time
        self.w_SDCM = self.w_SDCM + (newAlphai - self.alpha[i]) * self.y[i] * x
        self.b_SDCM = self.b_SDCM + (newAlphai - self.alpha[i]) * self.y[i]
        
        return newAlphai

    # Get the primal and the dual CSVM objective values in order to plot convergence curves
    # This is required for the dual solver which optimizes the dual objective function
    def getCSVMPrimalDualObjVals(self):
        hingeLoss = np.maximum( 1 - np.multiply( (self.X.dot( self.w_SDCM ) + self.b_SDCM), self.y ), 0 )
        objPrimal = 0.5 * self.w_SDCM.dot( self.w_SDCM ) + self.C * np.sum(hingeLoss)
        # Recall that b is supposed to be treated as the last coordinate of w
        objDual = np.sum( self.alpha ) - 0.5 * np.square( np.linalg.norm( self.w_SDCM ) ) - 0.5 * self.b_SDCM * self.b_SDCM
        
        return np.array( [objPrimal, objDual] )

    # Given a coordinate update oracle and a coordinate selection oracle, implement coordinate methods
    # The method returns the final model and the objective value acheived by the intermediate models
    # This can be used to implement coordinate descent or ascent as well as coordinate minimization or
    # maximization methods simply by modifying the coordinate update oracle
    def doSDCM(self, horizon = 10 ):
        objValSeries = []
        timeSeries = []
        totTime = 0
        selector = self.coordFunc[0] # Get hold of the function that will give me the next coordinate
        state = self.coordFunc[1] # The function needs an internal state variable - store the initial state
        
        for it in range( horizon ):
            # Start a stopwatch to calculate how much time we are spending
            tic = t.perf_counter()
            
            # Get the next coordinate to update and update that coordinate
            (i, state) = selector( state )
            self.alpha[i] = self.doCoordOptCSVMDual(i)

            toc = t.perf_counter()
            totTime = totTime + (toc - tic)
            
            objValSeries.append(self.getCSVMPrimalDualObjVals())
            timeSeries.append( totTime )
            
        return (self.alpha, objValSeries, timeSeries)
