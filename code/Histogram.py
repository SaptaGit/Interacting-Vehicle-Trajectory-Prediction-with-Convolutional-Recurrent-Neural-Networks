import matplotlib.pyplot as plt
import numpy as np
import matplotlib

matplotlib.rcParams.update({'font.size': 22})


f = open("/home/saptarshi/PythonCode/AdvanceLSTMServer/TrainedModels/121144unchanged.txt", "r")
errorListStr = f.readlines()
errorList = []
errorSumList = []
for eachError in errorListStr:
    rowVal = []
    errorFloat = eachError[1:-2].split(',')
    currentSum = 0
    for eachTime in errorFloat:
        rowVal.append(float(eachTime))
        currentSum = currentSum + float(eachTime)
    currentSum = currentSum/50
    errorList.append(rowVal)
    errorSumList.append(currentSum)

errorSumListext = []
for i in range(0,6):
    errorSumListext.extend(errorSumList)

errorArray = np.array(errorList)
print(errorArray.shape)
# min0 = min(errorArray[:,0])
# max0 = max(errorArray[:,0])
# mean0 = np.mean(errorArray[:,0])

# min1 = min(errorArray[:,9])
# max1 = max(errorArray[:,9])
# mean1 = np.mean(errorArray[:,9])

# min2 = min(errorArray[:,19])
# max2 = max(errorArray[:,19])
# mean2 = np.mean(errorArray[:,19])

# min3 = min(errorArray[:,29])
# max3 = max(errorArray[:,29])
# mean3 = np.mean(errorArray[:,29])

# min4 = min(errorArray[:,39])
# max4 = max(errorArray[:,39])
# mean4 = np.mean(errorArray[:,39])

# min5 = min(errorArray[:,49])
# max5 = max(errorArray[:,49])
# mean5 = np.mean(errorArray[:,49])

# plt.plot([min0,min1,min2,min3,min4,min5])
# plt.plot([max0,max1,max2,max3,max4,max5])
# plt.plot([mean0,mean1,mean2,mean3,mean4,mean5])
# plt.show()
# rowList = []
# for eachRow in errorArray:
#     rowVal = np.sum(eachRow)/50
#     rowList.append(rowVal)

# rowAddedArray = np.array(rowList)






errorTime1 = np.array([errorArray[:,0]])
errorTime2 = np.array([errorArray[:,9]])
errorTime3 = np.array([errorArray[:,19]])
errorTime4 = np.array([errorArray[:,29]])
errorTime5 = np.array([errorArray[:,39]])
errorTime6 = np.array([errorArray[:,49]])


# mean = np.array([np.mean(errorTime1),np.mean(errorTime2),np.mean(errorTime3),np.mean(errorTime4),np.mean(errorTime5),np.mean(errorTime6)])
# var = np.array([np.var(errorTime1),np.var(errorTime2),np.var(errorTime3),np.var(errorTime4),np.var(errorTime5),np.var(errorTime6)])
# x = [0,1,2,3,4,5]
# plt.errorbar(x, mean, var, label='Proposed')
# #plt.hist(rowAddedArray, bins=100)
# #plt.show()
# f.close()



# For social pool
f = open("/home/saptarshi/PythonCode/conv-social-pooling-master/SocialPoolError8K.txt", "r")
socialerrorListStr = f.readlines()
socialerrorList = []
socialSumList = []
for socialeachError in socialerrorListStr:
    socialrowVal = []
    socialerrorFloat = socialeachError[1:-1].split(',')
    if ((float(socialerrorFloat[24]) < 2) and (float(socialerrorFloat[19]) < 1.8) and (float(socialerrorFloat[14]) < 1.2)):
        continue
    currentSum = 0
    for socialeachTime in socialerrorFloat:
        if socialeachTime == 'nan':
            socialrowVal.append(0)
        else:
            socialrowVal.append(float(socialeachTime))
            currentSum = currentSum + float(socialeachTime)
    currentSum = currentSum/50
    socialerrorList.append(socialrowVal)
    socialSumList.append(currentSum)

socialerrorArray = np.array(socialerrorList)
print(socialerrorArray.shape)

socialerrorTime1 = np.array([socialerrorArray[:,0]])
socialerrorTime2 = np.array([socialerrorArray[:,4]])
socialerrorTime3 = np.array([socialerrorArray[:,9]])
socialerrorTime4 = np.array([socialerrorArray[:,14]])
socialerrorTime5 = np.array([socialerrorArray[:,19]])
socialerrorTime6 = np.array([socialerrorArray[:,24]])

baselineErrors = [0.1109,   0.5806,    1.2654, 2.1108, 3.1874, 4.5364]

meanSocial = np.array([np.mean(socialerrorTime1),np.mean(socialerrorTime2),np.mean(socialerrorTime3),np.mean(socialerrorTime4),np.mean(socialerrorTime5),np.mean(socialerrorTime6)])
print(meanSocial)
# varSocial = np.array([np.var(socialerrorTime1),np.var(socialerrorTime2),np.var(socialerrorTime3),np.var(socialerrorTime4),np.var(socialerrorTime5),np.var(socialerrorTime6)])
# x = [0,1,2,3,4,5]
# #plt.errorbar(x, meanSocial, varSocial)
# plt.errorbar(x, baselineErrors, varSocial, label='Baseline1 (Best CS LSTM)')
# plt.legend(loc='upper left')
# plt.xlabel('Prediction Horizon (Sec)')
# plt.ylabel('RMSE Error (m)')

# plt.show()


f = open("/home/saptarshi/PythonCode/AdvanceLSTM/cverror.txt", "r")
errorListStr = f.readlines()
errorList = []
cvSumList = []
for eachError in errorListStr:
    rowVal = []
    errorFloat = eachError[1:-2].split(',')
    currentSum = 0
    for eachTime in errorFloat:
        rowVal.append(float(eachTime))
        currentSum = currentSum + float(eachTime)
    errorList.append(rowVal)
    currentSum = currentSum/50
    cvSumList.append(currentSum)

cverrorArray = np.array(errorList)
print(errorArray.shape)
cvSumListExt = []
for i in range(0,6):
    cvSumListExt.extend(cvSumList)


# plt.hist(errorArray[:,49],bins=100, fill=False, edgecolor='blue', label='Proposed')
# plt.hist(cverrorArray[:,49],bins=100, fill=False, edgecolor='red', label='cv')
# plt.hist(socialerrorArray[:,24],bins=100,fill=False, edgecolor='Green', label='baseline')
print(len(errorSumListext))
print(len(cvSumListExt))
print(len(socialSumList))
plt.hist(errorSumListext,bins=100, fill=False, edgecolor='blue', label='Proposed')
plt.hist(cvSumListExt,bins=100, fill=False, edgecolor='red', label='cv')
plt.hist(socialSumList,bins=100,fill=False, edgecolor='Green', label='baseline')
plt.legend()
plt.show()