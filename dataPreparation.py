import numpy as np
from matplotlib import pyplot as plt
from scipy.io import loadmat
import pandas as pd
from scipy import stats

cepPiva = loadmat('projektR/Mjerenja/CrvenoTlo/cepPiva.mat')["cepPiva"]
clutter = loadmat('projektR/Mjerenja/CrvenoTlo/clutter.mat')["clutter"]
kuglica = loadmat('projektR/Mjerenja/CrvenoTlo/kuglica.mat')["kuglica"]
lipa20 = loadmat('projektR/Mjerenja/CrvenoTlo/lipa20.mat')["lipa20"]
metak = loadmat('projektR/Mjerenja/CrvenoTlo/metak.mat')["metak"]
PMA1 = loadmat('projektR/Mjerenja/CrvenoTlo/PMA1.mat')["PMA1"]
PMA2 = loadmat('projektR/Mjerenja/CrvenoTlo/PMA2.mat')["PMA2"]


def axesFunctions(newFieldName, title, axis, t, column):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle(title)
    fig.set_size_inches(20, 10)
    flag = 0
    for i in newFieldName[axis]:
        ax1.plot(i, "b")
        A = i[0]
        B = (np.log(i[column]/A)/column)
        if B > 0:
            newFieldName = np.delete(newFieldName, flag, 1)
        e = A * np.exp(B * t)
        ax2.plot(e, "b")
        flag += 1
    return newFieldName


def rearrange(fieldName, newFieldName, title, t, column):
    for i in range(len(fieldName)):
        axis = []
        for j in range(len(fieldName[i][0])):
            measurement = []
            for k in range(len(fieldName[i])):
                measurement.append(fieldName[i][k][j])
            axis.append(measurement)
        newFieldName.append(axis)
    newFieldName = axesFunctions(newFieldName, title + ' x', 0, t, column)
    newFieldName = axesFunctions(newFieldName, title + ' y', 1, t, column)
    newFieldName = axesFunctions(newFieldName, title + ' z', 2, t, column)
    return newFieldName


def allAxes(newFieldName, column):
    A = np.mean(newFieldName, 1)[0][0]
    B = np.log(np.mean(newFieldName, 1)[0][column]/A)/column
    e = A * np.exp(t * B)
    plt.figure(figsize=(10, 7))
    plt.plot(e, "y")
    plt.plot(np.mean(newFieldName, 1)[0], "r")
    plt.plot(np.mean(newFieldName, 1)[1], "g")
    plt.plot(np.mean(newFieldName, 1)[2], "b")
    plt.show()


def checkIfDeletionIsNeeded(newFieldName, x, y, z):
    for i in range(len(newFieldName[2])):
        if newFieldName[0][i][0] > x:
            print(i)
        if newFieldName[1][i][0] > y:
            print(i)
        if newFieldName[2][i][0] > z:
            print(i)


def reduceDimensionality(newFieldName, name, column):
    reducedField = []
    for i in newFieldName:
        axis = []
        for j in i:
            AandB = [j[0], np.log(j[column]/j[0])/column]
            axis.append(AandB)
        reducedField.append(axis)

    data[name] = reducedField


def concat(subdata):
    helper = []
    for i in range(len(subdata[0])):
        helper.append(np.concatenate(
            (subdata[0][i], subdata[1][i], subdata[2][i])))
    return helper


def zScore(df):
    df = df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]
    return df


data = {}
t = np.arange(94)

cepPivaRearranged = []
cepPivaRearranged = rearrange(cepPiva, cepPivaRearranged, 'Cep piva', t, 24)
#checkIfDeletionIsNeeded(cepPivaRearranged, 3, 100, 25)
cepPivaRearranged = np.delete(cepPivaRearranged, [54, 112], 1)

clutterRearranged = []
clutterRearranged = rearrange(clutter, clutterRearranged, 'Clutter', t, 24)
#checkIfDeletionIsNeeded(clutterRearranged, 3, 100, 5)
clutterRearranged = np.delete(clutterRearranged, 33, 1)

kuglicaRearranged = []
kuglicaRearranged = rearrange(kuglica, kuglicaRearranged, 'Kuglica', t, 24)
#checkIfDeletionIsNeeded(kuglicaRearranged, 0.5, 0.3, 1)
kuglicaRearranged = np.delete(kuglicaRearranged, 33, 1)

lipa20Rearranged = []
lipa20Rearranged = rearrange(lipa20, lipa20Rearranged, 'Lipa20', t, 24)
#checkIfDeletionIsNeeded(lipa20Rearranged, 0.64, 1, 1.25)
lipa20Rearranged = np.delete(
    lipa20Rearranged, [7, 9, 14, 25, 30, 113, 114, 116, 117, 133], 1)

metakRearranged = []
metakRearranged = rearrange(metak, metakRearranged, 'Metak', t, 24)
#checkIfDeletionIsNeeded(metakRearranged, 0.2, 0.3, 0.3)
metakRearranged = np.delete(metakRearranged, [29, 93, 95, 97, 101, 104], 1)

PMA1Rearranged = []
PMA1Rearranged = rearrange(PMA1, PMA1Rearranged, 'PMA1', t, 24)
#checkIfDeletionIsNeeded(PMA1Rearranged, 0.37, 0.4, 0.5)
PMA1Rearranged = np.delete(
    PMA1Rearranged, [17, 21, 23, 24, 26, 27, 32, 35, 37, 39, 41, 43, 134], 1)

PMA2Rearranged = []
PMA2Rearranged = rearrange(PMA2, PMA2Rearranged, 'PMA2', t, 24)
#checkIfDeletionIsNeeded(PMA2Rearranged, 25000, 0.3, 2.5)
PMA2Rearranged = np.delete(PMA2Rearranged, [76, 80, 82, 144], 1)

reduceDimensionality(cepPivaRearranged, "Cep piva", 24)
reduceDimensionality(clutterRearranged, "Clutter", 24)
reduceDimensionality(kuglicaRearranged, "Kuglica", 24)
reduceDimensionality(lipa20Rearranged, "Lipa 20", 24)
reduceDimensionality(metakRearranged, "Metak", 24)
reduceDimensionality(PMA1Rearranged, "PMA1", 24)
reduceDimensionality(PMA2Rearranged, "PMA2", 24)

for i in data:
    data[i] = concat(data[i])

binaryClassificationData = []
multiClassificationData = []
counter = 0
for i in data:
    for j in range(len(data[i])):
        if (i == "PMA1" or i == "PMA2"):
            binaryClassificationData.append(np.concatenate((data[i][j], [1])))
        else:
            binaryClassificationData.append(np.concatenate((data[i][j], [0])))
        multiClassificationData.append(np.concatenate((data[i][j], [counter])))
    counter += 1

binaryClassificationData = pd.DataFrame(binaryClassificationData, columns=[
                                        "Ax", "Bx", "Ay", "By", "Az", "Bz", "label"])
multiClassificationData = pd.DataFrame(multiClassificationData, columns=[
                                       "Ax", "Bx", "Ay", "By", "Az", "Bz", "label"])

#np.savetxt(r'C:\PROJEKT_R\projektR\binaryClassification.txt', binaryClassificationData.values, fmt='%.4f')
#np.savetxt(r'C:\PROJEKT_R\projektR\multiClassification.txt', multiClassificationData.values, fmt='%.4f')

binaryClassificationDataWithSummedAxes = pd.DataFrame()
binaryClassificationDataWithSummedAxes["A sum"] = binaryClassificationData["Ax"] + \
    binaryClassificationData["Ay"] + binaryClassificationData["Az"]

binaryClassificationDataWithSummedAxes["B sum"] = binaryClassificationData["Bx"] + \
    binaryClassificationData["By"] + binaryClassificationData["Bz"]

binaryClassificationDataWithSummedAxes["label"] = binaryClassificationData["label"]
#np.savetxt(r'C:\PROJEKT_R\projektR\binaryClassificationWithSummedAxes.txt', binaryClassificationDataWithSummedAxes.values, fmt='%.4f')

binaryClassificationDataWithMultipliedAxes = pd.DataFrame()
binaryClassificationDataWithMultipliedAxes["A mul"] = binaryClassificationData["Ax"] * \
    binaryClassificationData["Ay"] * binaryClassificationData["Az"]

binaryClassificationDataWithMultipliedAxes["B mul"] = binaryClassificationData["Bx"] * \
    binaryClassificationData["By"] * binaryClassificationData["Bz"]

binaryClassificationDataWithMultipliedAxes["label"] = binaryClassificationData["label"]
#np.savetxt(r'C:\PROJEKT_R\projektR\binaryClassificationWithMultipliedAxes.txt', binaryClassificationDataWithMultipliedAxes, fmt='%.4f')
