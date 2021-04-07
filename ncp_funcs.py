#!/usr/bin.env python

'''
ncp_funcs.py
Created by: joshthomastaylor
On: 06/04/2021

'''

import numpy as np 
import matplotlib.pyplot as plt 
import random
import ncp_params
import datetime
import time
import os


'''
functions for dm_test.py
'''

def genObj(Ox, Op, Px):
	'''
	Generate a zero-padded object
	:return: nxn array
	'''
	obj = np.zeros([Ox, Ox])
	randPos = random.sample(range(0, np.square(Ox)-1), int(np.square(Ox)*Op))

	#change the value of indexed (randPos) positions to a value between 0 and 1
	for i in randPos:
		x, y = divmod(i, Ox)
		obj[x, y] = np.random.rand(1)

	#zero pad the object
	padObj = np.zeros([Px, Px])
	padObj[Ox//2:Px-Ox//2, Ox//2:Px-Ox//2] = obj
	return(padObj)



def projFour(itr, conFour):
	'''
	Project iterate onto Fourier constraint
	:param itr: current iterate
	:param conFour: the Fourier amplitudes of true object
	:return: iterate with adjusted Fourier amplitudes
	'''
	mag = conFour
	phase = np.angle(np.fft.fftn(itr))
	z = mag*np.exp(1j*phase)
	return(np.real(np.fft.ifftn(z)))



def projSupp(itr, conSupp):
	'''
	Project iterate onto support constraint
	:param itr: current iterate
	:param conSupp: the support constraint of true object
	:return: iterate with adjusted support
	'''
	return(itr*conSupp)

def r_projFour(itr, conFour, beta):
	'''
	Relaxed Fourier projection: linear combination
	:param itr: current iterate
	:param conFour: the Fourier amplitudes of true object
	:param beta: parameter
	:return: iterate with adjusted Fourier amplitudes
	'''
	gamA = -1/beta
	pA = projFour(itr, conFour)
	return(pA+gamA*(pA-itr))


def r_projSupp(itr, conSupp, beta):
	gamB = 1/beta
	pB = projSupp(itr, conSupp)
	return(pB+gamB*(pB-itr))


def algDM(itr, conFour, conSupp, beta):
	x1 = projFour(r_projSupp(itr, conSupp, beta), conFour)
	x2 = projSupp(r_projFour(itr, conFour, beta), conSupp)
	itr = itr+beta*(x1-x2)
	return(itr, x2)

'''
functions for dm_mask.py
'''

def genSupps(Ox, Op, Px, Sn):
	'''
	Generate n zero-padded supports
	:return: nxn array
	'''
	supps = []

	for i in range(Sn):
		obj = np.zeros([Ox, Ox])
		randPos = random.sample(range(0, np.square(Ox)-1), int(np.square(Ox)*Op))

		#change the value of indexed (randPos) positions to a value between 0 and 1
		for j in randPos:
			x, y = divmod(j, Ox)
			obj[x, y] = 1

		#zero pad the object
		padObj = np.zeros([Px, Px])
		padObj[Ox//2:Px-Ox//2, Ox//2:Px-Ox//2] = obj

		supps.append(padObj)
	return(supps)

def errRMS(arg1, arg2):
	a = np.sum(np.square(arg2 - arg1))
	b = np.sum(np.square(arg1))
	return(np.sqrt(a/b))

def errMetrics(trueObj, conFour, itrSoln):
	errFour = errRMS(conFour, np.abs(np.fft.fftn(itrSoln)))
	errReal = errRMS(trueObj, itrSoln)
	return(errFour, errReal)

def projSupp_mask(itr, supps):
	errArr = []
	for i in supps:
		test = itr*i
		error = errRMS(itr, test)
		errArr.append(error)

	errMin = errArr.index(min(errArr))
	ncp_params.suppSelect.append(errMin)
	return(itr*supps[errMin])

def r_projSupp_mask(itr, supps, beta):
	gamB = 1/beta
	pB = projSupp_mask(itr, supps)
	return(pB+gamB*(pB-itr))


def algDM_mask(itr, conFour, supps, beta):
	x1 = projFour(r_projSupp_mask(itr, supps, beta), conFour)
	x2 = projSupp_mask(r_projFour(itr, conFour, beta), supps)
	itr = itr+beta*(x1-x2)
	return(itr, x2)

def arrMerge(a, b, c, d, e, f, g):
	m = [a, b, c, d, e, f]
	return(m+g)

def arrSave(arrAll, elapsedT):
	if not os.path.exists("../../results/suppMask"+str(ncp_params.Sn)):
		os.mkdir("../../results/suppMask"+str(ncp_params.Sn))

	nameRun = str(datetime.datetime.now())
	nameRun = nameRun.replace(':', '')

	nameFolder = "../../results/suppMask"+str(ncp_params.Sn)+"/"+nameRun
	os.mkdir(nameFolder)

	arrIds =  ["%.2d" % i for i in range(len(arrAll))]
	for i in range(len(arrAll)):
		nameFile = nameFolder+"/"+arrIds[i]+".csv"
		np.savetxt(nameFile, arrAll[i], delimiter = ',')

	#create .txt file with extra details
	txtTitle = np.array(['Iterations:', 'Beta:', 'Time:', 'Supports:'])
	txtData = np.array([ncp_params.N, ncp_params.beta, elapsedT, ncp_params.Sn])
	txtArr = np.zeros(txtTitle.size, dtype=[('var1', 'U6'), ('var2', float)])
	txtArr['var1'] = txtTitle
	txtArr['var2'] = txtData
	np.savetxt(nameFolder+"/"+'info.txt', txtArr, fmt="%s %10.3f")

	return()

'''
functions for dm_mask_REF.py
'''

def projFourREF(itr, conFour):
	z = conFour + np.imag(np.fft.fftn(itr))*1j
	return(np.real(np.fft.ifftn(z)))

def r_projFourREF(itr, conFour, beta):
	gamA = -1/beta
	pA = projFourREF(itr, conFour)
	return(pA+gamA*(pA-itr))

def errMetricsREF(trueObj, conFour, itrSoln):
	errFour = errRMS(conFour, np.real(np.fft.fftn(itrSoln)))
	errReal = errRMS(trueObj, itrSoln)
	return(errFour, errReal)

def algDM_maskREF(itr, conFour, supps, beta):
	x1 = projFourREF(r_projSupp_mask(itr, supps, beta), conFour)
	x2 = projSupp_mask(r_projFourREF(itr, conFour, beta), supps)
	itr = itr+beta*(x1-x2)
	return(itr, x2)