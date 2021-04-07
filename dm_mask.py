#!/usr/bin.env python

'''
dm_mask.py
Created by: joshthomastaylor
On: 07/04/2021

'''

import numpy as np 
import matplotlib.pyplot as plt 
import random 
import ncp_params
import ncp_funcs
import time


startT = time.perf_counter()


#initialise
Ox, Op, Px, Sn = ncp_params.Ox, ncp_params.Op, ncp_params.Px, ncp_params.Sn
trueObj = ncp_funcs.genObj(Ox, Op, Px)
initObj = ncp_funcs.genObj(Ox, Op, Px)
supps = ncp_funcs.genSupps(Ox, Op, Px, Sn)

itr = initObj.copy()

#define support(s) constraint
conSupp = trueObj.copy()
conSupp[conSupp>0] = 1

########################
#insert correct support
supps.insert(0, conSupp)
########################

#define Fourier constraint
conFour = np.abs(np.fft.fftn(trueObj))


#error metrics
errFour = np.zeros(ncp_params.N)
errReal = np.zeros(ncp_params.N)
ncp_params.suppSelect = []



#iterate!
for i in range(ncp_params.N):

	itr, itrSoln = ncp_funcs.algDM_mask(itr, conFour, supps, ncp_params.beta)

	errFour[i], errReal[i] = ncp_funcs.errMetrics(trueObj, conFour, itrSoln)


endT = time.perf_counter()
elapsedT = endT - startT

#save run
arrAll = ncp_funcs.arrMerge(initObj, itrSoln, trueObj, errFour, errReal, ncp_params.suppSelect[1::2], supps)
ncp_funcs.arrSave(arrAll, elapsedT)


#display!
axes=[]
fig=plt.figure()
axes.append(fig.add_subplot(1, 2, 1) ) 
plt.imshow(arrAll[1])
axes.append(fig.add_subplot(1,2, 2) ) 
plt.imshow(arrAll[2])
fig.tight_layout()    
plt.show()

plt.plot(range(0, ncp_params.N), arrAll[3])
plt.show()






