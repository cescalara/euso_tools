import ROOT
import numpy as np 
#import pandas as pd
import sys
from smooth import smooth
from matplotlib import pyplot as plt
from matplotlib.pyplot import ion
from rootpy.plotting import root2matplotlib as rplt 
from rootpy.plotting import Hist2D
from rootpy.plotting import Hist1D
from matplotlib import colors, ticker, cm
from matplotlib.mlab import find
from scipy.signal import lfilter, lfilter_zi, filtfilt, butter, kaiser, decimate, correlate
from numpy.fft import rfft

from ipywidgets import FloatProgress
from IPython.display import display


def analysis(filename):
	"""
	Main analysis program to extract data in numpy array and calculate smoothed signals from data
	Also code for finding percentages of frames above some threshold value (eg. mean)
	"""
	ion()

	packet_size = 128


	#Open file
	f = ROOT.TFile(filename, "r")

	#Draw photon counts for a certain pixel
	#f.tevent.Draw("photon_count_data[0][0][35][35]:Entry$", "Entry$<100000", "*")

	N = f.tevent.GetEntries()
	# Set the TTree photon count data array to a python array - not needed for a 1D TTree entries, but needed for multidimensional
	pcd_tmp = np.zeros((1, 1, 48, 48), dtype='B')
	gtu_time = np.zeros((1,1))
	f.tevent.SetBranchAddress("photon_count_data", pcd_tmp)
	f.tevent.SetBranchAddress("gtu_time", gtu_time)



	# Read all the data from a specified GTU into memory
	L = []
	S = []
	gt = []
	GTU = range(N)
	pcd_tot = []

	for i in GTU:
		f.tevent.GetEntry(i)
		pcd = pcd_tmp[0][0][:][:]
		pcd_tot = pcd_tot.append(pcd)
		L.append(np.mean(pcd))
		S.append(np.sum(pcd))
		gt.append(gtu_time[0,0])
		print i

	#pcd_tot = np.dstack(L)
	#print L

	#plt.plot(GTU[1:3000],L[1:3000])

	#Smooth the signal
	#rm5 = pd.rolling_mean(np.array(L),50)
	#s = smooth(np.array(L))
	"""
	#Optimised implementation, forward and backward implementation of IIR filter
	b, a = butter(3, 0.05)
	ff = filtfilt(b, a, np.array(L))
	#plt.plot(GTU[1:3000],ff[1:3000])

	#Calculate percent of frames with signal > mean(L)
	perc_gt_mean = (np.sum(np.array(L) > np.mean(L)))*100/np.size(L)

	#Test some other values, ie. signal > thres
	thres = np.array([0.10, 0.125, 0.15, 0.175, 0.2])
	p = []
	for j in thres:
		p.append(np.sum(np.array(L) > j)*100/np.size(L))


	#Split data into packets of size 128GTUs and calc perc above mean signal for each
	#J = np.delete(np.array(L), 320000)
	#packets = np.split(np.array(L), np.size(L)/packet_size)
	#packets = np.array(packets)
	#p = []

	#for k in range(np.size(L)/packet_size):
	#	p.append(np.sum(packets[k][:] > np.mean(L))*100/np.size(packets[k][:]))
	#	print(k)
	"""

	"""
	rep = ''
	while not rep in [ 'q', 'Q' ]:
		rep = raw_input('enter "q" to quit: ')
		if 1 < len(rep):
			rep = rep[0]
	"""
	gt=np.array(gt)
	return pcd_tot, L, S, gt

def ECanalysis(filename):
	"""
	Analysis as above function but for each EC separately 
	"""
	ion()

	packet_size = 128

	#Open file
	f = ROOT.TFile(filename, "r")

	N = f.tevent.GetEntries()
	# Set the TTree data to python array
	pcd_tmp = np.zeros((1, 1, 48, 48), dtype='B')
	gtu_time = np.zeros((1,1))
	f.tevent.SetBranchAddress("photon_count_data", pcd_tmp)
	f.tevent.SetBranchAddress("gtu_time", gtu_time)

	# Read all the data from a specified GTU into memory
	gt = []
	GTU = range(N)
	ECnum = range(9)
	EC = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [] , 8: []}

	for i in GTU:
		f.tevent.GetEntry(i)
		pcd = pcd_tmp[0][0][:][:]
		gt.append(gtu_time[0,0])
		pcd[pcd>100] = 0 # to get rid of crazy pixel
		pcd_splt = blockshaped(pcd,16,16)
		for j in ECnum: 
			EC[j].append(np.sum(pcd_splt[j][:][:]))	 
		print i
	"""
	#smooth the signal
	EC_filt = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [] , 8: []}
	for k in ECnum:
		EC_filt[k] = pd.rolling_mean(np.array(EC[k]), 50)
	"""
	#Reset gtu time to 0
	gt=gt-gt[0]
	gt=np.array(gt)

	return EC, gt

def PMTanalysis(filename):
	"""
	Analysis as above function but for each PMT separately 
	"""
	ion()

	packet_size = 128

	#Open file
	f = ROOT.TFile(filename, "r")

	N = f.tevent.GetEntries()
	# Set the TTree data to python array
	pcd_tmp = np.zeros((1, 1, 48, 48), dtype='B')
	gtu_time = np.zeros((1,1))
	f.tevent.SetBranchAddress("photon_count_data", pcd_tmp)
	f.tevent.SetBranchAddress("gtu_time", gtu_time)

	# Read all the data from a specified GTU into memory
	gt = []
	GTU = range(N)
	PMTnum = range(36)
	PMT = {0:  [], 1:  [], 2:  [], 3:  [], 4:  [], 5:  [], 6:  [], 7:  [],  8: [],
		  9:  [], 10: [], 11: [], 12: [], 13: [], 14: [], 15: [], 16: [], 17: [],
		  18: [], 19: [], 20: [], 21: [], 22: [], 23: [], 24: [], 25: [], 26: [],
		  27: [], 28: [], 29: [], 30: [], 31: [], 32: [], 33: [], 34: [], 35: []}

        prog = FloatProgress(min = 0, max = N)
        display(prog)

        for i in range(N):
		f.tevent.GetEntry(i)
		pcd = pcd_tmp[0][0][:][:]
		gt.append(gtu_time[0,0])
		pcd[pcd>100] = 0 # to get rid of crazy pixel
		pcd_splt = blockshaped(pcd,8,8)
		for j in PMTnum: 
			PMT[j].append(np.sum(pcd_splt[j][:][:]))	 
		#print i
                prog.value+=1

	"""
	#smooth the signal
	PMT_filt = {0:  [], 1:  [], 2:  [], 3:  [], 4:  [], 5:  [], 6:  [], 7:  [],  8: [],
		  9:  [], 10: [], 11: [], 12: [], 13: [], 14: [], 15: [], 16: [], 17: [],
		  18: [], 19: [], 20: [], 21: [], 22: [], 23: [], 24: [], 25: [], 26: [],
		  27: [], 28: [], 29: [], 30: [], 31: [], 32: [], 33: [], 34: [], 35: []}
	for k in PMTnum:
		PMT_filt[k] = pd.rolling_mean(np.array(PMT[k]), 50)
	"""
	#Reset gtu time to 0
	gt=gt-gt[0]
	gt=np.array(gt)
	

	return PMT, gt

def blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))


def parabolic(f, x):
    """
    Quadratic interpolation for estimating the true position of an
    inter-sample maximum when nearby samples are known.
    
    f is a vector and x is an index for that vector.
    
    Returns (vx, vy), the coordinates of the vertex of a parabola that goes
    through point x and its two neighbors.
    """
    xv = 1/2. * (f[x-1] - f[x+1]) / (f[x-1] - 2 * f[x] + f[x+1]) + x
    yv = f[x] - 1/4. * (f[x-1] - f[x+1]) * (xv - x)
    return (xv, yv)


def freq_from_fft(signal, fs):
    """
    Estimate frequency from peak of FFT - doesn't seem to work well, lost in harmonics?
    """
    N = len(signal)
    
    # Compute Fourier transform of windowed signal
    windowed = signal * kaiser(N, 100)
    f = rfft(windowed)
    # Find the peak and interpolate to get a more accurate peak
    i_peak = np.argmax(abs(f)) # Just use this value for less-accurate result
    i_interp = parabolic(np.log(abs(f)), i_peak)[0]
    
    # Convert to equivalent frequency
    return fs * i_interp / N # Hz


def freq_from_hps(signal, fs):
    """
    Estimate frequency using harmonic product spectrum - also not working so well
    Low frequency noise piles up and overwhelms the peaks?
    """
    N = len(signal)
    signal -= np.mean(signal) # Remove DC offset
    
    # Compute Fourier transform of windowed signal
    windowed = signal * kaiser(N, 100)
    
    # Get spectrum
    X = np.log(abs(rfft(windowed)))
    
    # Downsample sum logs of spectra instead of multiplying
    hps = np.copy(X)
    for h in np.arange(2, 9): # TODO: choose a smarter upper limit
        dec = decimate(X, h)
        hps[:len(dec)] += dec
    
    # Find the peak and interpolate to get a more accurate peak
    i_peak = np.argmax(hps[:len(dec)])
    i_interp = parabolic(hps, i_peak)[0]
    
    # Convert to equivalent frequency
    return fs * i_interp / N # Hz

def freq_from_crossings(signal, fs):
    """
    Estimate frequency by counting rising-edge zero crossings
    Seems to work well as long as signal is long
    """
    # Remove offset from zero
    signal -= np.mean(signal)

    # Find all indices right before a rising-edge zero crossing
    indices = find((signal[1:] >= 0) & (signal[:-1] < 0))
    
    # Shortcut, less accurtae
    #crossings = indices
    
    # More accurate, using linear interpolation to find intersample
    # zero-crossings (Measures 1000.000129 Hz for 1000 Hz, for instance)
    crossings = [i - signal[i] / (signal[i+1] - signal[i]) for i in indices]
    
    return fs / np.mean(np.diff(crossings))

def find_time_diff(dark_signal, sky_signal):
	""" 
	Find phase shift between two signals using the cross correlation 

	A <=> dark signal
	B <=> sky signal
	"""

	#Get np arrays from input lists
	A = np.array(dark_signal)
	B = np.array(sky_signal)


	# Find time shift between signals using the correlation
	#nsamples = 320000
	#A = A[0:nsamples]
	#B = B[0:nsamples]

	nsamples=np.size(A)

	# regularize datasets by subtracting mean and dividing by s.d.
	A -= np.mean(A); A /= np.std(A)
	B -= np.mean(B); B /= np.std(B)

	# Put in an artificial time shift between the two datasets to check
	#time_shift = 0
	#A = np.roll(A, time_shift)

	# Find cross-correlation
	xcorr = correlate(A, B)

	# delta time array to match xcorr
	dt = np.arange(1-nsamples, nsamples)

	recovered_time_shift = dt[np.argmax(xcorr)]

	#print "Added time shift: %d" % (time_shift) #Use when testing
	print "Recovered time shift: %d" % (recovered_time_shift)

	return recovered_time_shift, xcorr 

def plot_packets(signal, N):
	"""
	Plot signal in packets of length N
	"""
	L = np.float(len(signal))
	if (L/N).is_integer(): 
		packets = np.split(signal, L/N)
		packets = np.array(packets)
	else:
		floor = np.floor(L/N)
		signal = signal[0:floor*N] #cut signal down to largest possible size
		L_new = len(signal)
		packets = np.split(signal,L_new/N)


	for i in range(np.shape(packets)[0]):
		plt.plot(packets[:][i])

	return packets

def split_into_packets(signal, N):
	"""
	Split signal into packets of length N. 
	"""

	L = np.float(len(signal))
	if (L/N).is_integer(): 
		packets = np.split(signal, L/N)
		packets = np.array(packets)
	else:
		floor = np.floor(L/N)
		signal = signal[0:floor*N] #cut signal down to largest possible size
		L_new = len(signal)
		packets = np.split(signal,L_new/N)

	return packets

#def rm_para_sig(signal, filt_sig):
	"""
	Remove the parasitic signal via fitting of the dark signal via  cross-correlation of the digitally smoothed 
	dark signal with the input signal
	"""

















#########
# Here is a list of five 10x10 arrays:
#x=[np.random.random((10,10)) for _ in range(5)]

#y=np.dstack(x)
#print(y.shape)
# (10, 10, 5)

# To get the shape to be Nx10x10, you could  use rollaxis:
#y=np.rollaxis(y,-1)
#print(y.shape)
# (5, 10, 10)
