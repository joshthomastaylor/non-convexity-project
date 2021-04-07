#!/usr/bin.env python

'''
dm_test.py
Created by: joshthomastaylor
On: 07/04/2021

'''

import numpy as np 
import matplotlib.pyplot as plt 
import random 
import ncp_params
import ncp_funcs

Ox, Op, Px = ncp_params.Ox, ncp_params.Op, ncp_params.Px
trueObj = ncp_funcs.genObj(Ox, Op, Px)
initObj = ncp_funcs.genObj(Ox, Op, Px)

itr = initObj

#define support constraint
conSupp = trueObj.copy()
conSupp[conSupp>0] = 1

#define Fourier constraint
conFour = np.abs(np.fft.fftn(trueObj))

#iterate!
for i in range(ncp_params.N):

	itr, soln = ncp_funcs.algDM(itr, conFour, conSupp, ncp_params.beta)

#display!
axes=[]
fig=plt.figure()
axes.append(fig.add_subplot(1, 2, 1) ) 
plt.imshow(soln)
axes.append(fig.add_subplot(1,2, 2) ) 
plt.imshow(trueObj)
fig.tight_layout()    
plt.show()







