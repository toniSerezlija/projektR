from re import X
from numpy import array
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# curve used for fitting


def Gauss(x, A, B):

    if(A < 0):
        A = 0
    if(B < 0):
        B = 0
    y = A * np.exp(-1*B*x**2)
    return y


metalObject = input(
    "Unesite ime metalnog objekta za koji želite vidjeti mjerenja (cepPiva, clutter, kuglica, lipa20, metak, PMA1, PMA2): ")

fileName = "C:/PROJEKT_R/projektR/Mjerenja/CrvenoTlo/" + metalObject

COLUMNS = 94

mat = scipy.io.loadmat(fileName)
data = mat.get(metalObject)  # loading data from matlab

arrData = np.array(data)
xpoints = np.arange(COLUMNS)  # from 0 to 93 microseconds

xMjerenja = np.array(arrData[0])
yMjerenja = np.array(arrData[1])
zMjerenja = np.array(arrData[2])


testnoMjerenjeX = np.empty(COLUMNS)
testnoMjerenjeY = np.empty(COLUMNS)
testnoMjerenjeZ = np.empty(COLUMNS)

outliersX = np.empty(COLUMNS)
outliersY = np.empty(COLUMNS)
outliersZ = np.empty(COLUMNS)

najveciX = 0
najveciY = 0
najveciZ = 0

# number of measurements
zadnjeMjerenje = (int)(arrData.size / (COLUMNS * 3))

brojMjerenja = int(
    input("Unesite broj mjerenja za koje zelite vidjeti podatke, a koje je manje ili jednako od zadnjeg mjerenja (" + str(zadnjeMjerenje) + "): "))

if(brojMjerenja > zadnjeMjerenje):
    exit("Ne postoji takvo mjerenje")

# finding the biggest number in each array
for i in range(COLUMNS - 1):
    if(((xMjerenja[i])[brojMjerenja - 1]) > najveciX):
        najveciX = ((xMjerenja[i])[brojMjerenja - 1])
    if(((yMjerenja[i])[brojMjerenja - 1]) > najveciY):
        najveciY = ((yMjerenja[i])[brojMjerenja - 1])
    if(((zMjerenja[i])[brojMjerenja - 1]) > najveciZ):
        najveciZ = ((zMjerenja[i])[brojMjerenja - 1])

#loading and scaling
for i in range(COLUMNS):
    testnoMjerenjeX[i] = ((xMjerenja[i])[brojMjerenja - 1])
    testnoMjerenjeY[i] = ((yMjerenja[i])[brojMjerenja - 1])
    testnoMjerenjeZ[i] = ((zMjerenja[i])[brojMjerenja - 1])

# INTER QUARTILE RANGE
for i in range(COLUMNS - 1):
    outliersX[i] = testnoMjerenjeX[i]
    outliersY[i] = testnoMjerenjeY[i]
    outliersZ[i] = testnoMjerenjeZ[i]

# sorted arrays for iqr
outliersX.sort()
outliersY.sort()
outliersZ.sort()

# calculating first and third quantile
Q1_X = np.quantile(outliersX, 0.25)
Q3_X = np.quantile(outliersX, 0.75)

Q1_Y = np.quantile(outliersY, 0.25)
Q3_Y = np.quantile(outliersY, 0.75)

Q1_Z = np.quantile(outliersZ, 0.25)
Q3_Z = np.quantile(outliersZ, 0.75)

# calculating the IQR
IQR_X = Q3_X - Q1_X
IQR_Y = Q3_Y - Q1_Y
IQR_Z = Q3_Z - Q1_Z

MULTIPLIER = 1.5

upper_limitX = Q3_X + MULTIPLIER * IQR_X
upper_limitY = Q3_Y + MULTIPLIER * IQR_Y
upper_limitZ = Q3_Z + MULTIPLIER * IQR_Z

lower_limitX = Q1_X - MULTIPLIER * IQR_X
lower_limitY = Q3_Y - MULTIPLIER * IQR_Y
lower_limitZ = Q3_Z - MULTIPLIER * IQR_Z

brojacX = 0
brojacY = 0
brojacZ = 0

# finding size for new arrays without outliers
for i in range(COLUMNS - 1):
    if(testnoMjerenjeX[i] >= lower_limitX and testnoMjerenjeX[i] <= upper_limitX):
        brojacX += 1
    if(testnoMjerenjeY[i] >= lower_limitY and testnoMjerenjeY[i] <= upper_limitY):
        brojacY += 1
    if(testnoMjerenjeZ[i] >= lower_limitZ and testnoMjerenjeZ[i] <= upper_limitZ):
        brojacZ += 1

# initializing new arrays without outliers
newX = np.empty(brojacX)
newY = np.empty(brojacY)
newZ = np.empty(brojacZ)

# loading data in new arrays without outliers
pom = 0
for i in range(COLUMNS - 1):
    if(testnoMjerenjeX[i] >= lower_limitX and testnoMjerenjeX[i] <= upper_limitX):
        newX[pom] = testnoMjerenjeX[i]
        pom += 1

pom = 0
for i in range(COLUMNS - 1):
    if(testnoMjerenjeY[i] >= lower_limitY and testnoMjerenjeY[i] <= upper_limitY):
        newY[pom] = testnoMjerenjeY[i]
        pom += 1

pom = 0
for i in range(COLUMNS - 1):
    if(testnoMjerenjeZ[i] >= lower_limitZ and testnoMjerenjeZ[i] <= upper_limitZ):
        newZ[pom] = testnoMjerenjeZ[i]
        pom += 1


# CURVE FITTING
parametersX, covarianceX = curve_fit(Gauss, xpoints, testnoMjerenjeX)

fit_AX = parametersX[0]  # the best fit value of A for x axis
fit_BX = parametersX[1]  # the best fit value of B for x axis
fit_yX = Gauss(xpoints, fit_AX, fit_BX)

parametersY, covarianceY = curve_fit(Gauss, xpoints, testnoMjerenjeY)

fit_AY = parametersY[0]  # the best fit value of A for y axis
fit_BY = parametersY[1]  # the best fit value of B for y axis
fit_yY = Gauss(xpoints, fit_AY, fit_BY)

parametersZ, covarianceZ = curve_fit(Gauss, xpoints, testnoMjerenjeZ)

fit_AZ = parametersZ[0]  # the best fit value of A for z axis
fit_BZ = parametersZ[1]  # the best fit value of B for z axis
fit_yZ = Gauss(xpoints, fit_AZ, fit_BZ)

axis = input("Odaberite os za koju želite vidjeti mjerenje (x,y ili z): ")

if(axis.lower() == 'x'):
    plt.figure(100)
    plt.plot(xpoints, testnoMjerenjeX, 'o', label='x data')
    plt.plot(xpoints, fit_yX, '-', label='fitX')
    plt.legend()
    plt.figure(200)
    sns.distplot(testnoMjerenjeX, label='with outliers')
    plt.legend()
    plt.figure(300)
    sns.distplot(newX, label='without outliers')
    plt.legend()
elif(axis.lower() == 'y'):
    plt.figure(100)
    plt.plot(xpoints, testnoMjerenjeY, 'o', label='y data')
    plt.plot(xpoints, fit_yY, '-', label='fitY')
    plt.legend()
    plt.figure(200)
    sns.distplot(testnoMjerenjeY, label='with outliers')
    plt.legend()
    plt.figure(300)
    sns.distplot(newY, label='without outliers')
    plt.legend()
elif(axis.lower() == 'z'):
    plt.figure(100)
    plt.plot(xpoints, testnoMjerenjeZ, 'o', label='z data')
    plt.plot(xpoints, fit_yZ, '-', label='fitZ')
    plt.legend()
    plt.figure(200)
    sns.distplot(testnoMjerenjeZ, label='with outliers')
    plt.legend()
    plt.figure(300)
    sns.distplot(newZ, label='without outliers')
    plt.legend()
else:
    exit("Not an axis")

plt.show()
