import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({'font.size': 16})

lbda = 1.
L = np.pi
chi0 = 1.
N = 150
M = 15
Marr = np.linspace(-(M-1)/2,(M-1)/2,M)

def chi(r):
	return chi0*np.exp(-r**2/(2*lbda**2))

def phi(r):
	return sum(chi(r + Marr * L))

Xarr = np.linspace(-L,L,N)
CHIarr = chi(Xarr)
PHIarr = np.zeros(N)
for i in range(N):
	PHIarr[i] = phi(Xarr[i])

plt.figure
plt.plot(Xarr,CHIarr, label = r'$\chi$')
plt.plot(Xarr,PHIarr, label = r'$\varphi$')
plt.xlabel(r'$x$')
plt.legend(loc='best')
plt.show()
