import numpy as np
import pandas as pd

maxIter = 500
tol = 10**-5
modelNumber = 5
classNumber = 10

## READING TRAINING DATA
xTrainInput = pd.read_csv('X_train.csv',header=None,nrows=60000).as_matrix()
yTrainOutput = pd.read_csv('y_train.csv',header=None,nrows=60000).as_matrix()

dataPointsTrain = xTrainInput.shape[0]
dimensions = xTrainInput.shape[1]

xTrainClass0 = None
xTrainClass1 = None
xTrainClass2 = None
xTrainClass3 = None
xTrainClass4 = None
xTrainClass5 = None
xTrainClass6 = None
xTrainClass7 = None
xTrainClass8 = None
xTrainClass9 = None

xtrainDotClass0 = None
xtrainDotClass1 = None
xtrainDotClass2 = None
xtrainDotClass3 = None
xtrainDotClass4 = None
xtrainDotClass5 = None
xtrainDotClass6 = None
xtrainDotClass7 = None
xtrainDotClass8 = None
xtrainDotClass9 = None

## PARTITIONING INTO 10 CLASSES
print("PARTITIONING TRAINING DATA")
classRatioVectorTraining = np.zeros((10,1),dtype = float)
for z in range(xTrainInput.shape[0]):
    copyXVector = np.reshape(np.copy(xTrainInput[z]),(1, dimensions))
    xDotProduct = np.reshape(np.dot(copyXVector,np.transpose(copyXVector)),(1,1))
    
    if yTrainOutput[z] == 0:
        classRatioVectorTraining[0]+=1
        if xTrainClass0 is None:
            xTrainClass0 = copyXVector
            xtrainDotClass0 = xDotProduct
        else:
            xTrainClass0 = np.concatenate((xTrainClass0,copyXVector), axis = 0)
            xtrainDotClass0 = np.concatenate((xtrainDotClass0,xDotProduct),axis = 0)
            
    if yTrainOutput[z] == 1:
        classRatioVectorTraining[1]+=1
        if xTrainClass1 is None:
            xTrainClass1 = copyXVector
            xtrainDotClass1 = xDotProduct
        else:
            xTrainClass1 = np.concatenate((xTrainClass1,copyXVector), axis = 0)
            xtrainDotClass1 = np.concatenate((xtrainDotClass1,xDotProduct),axis=0)
            
    if yTrainOutput[z] == 2:
        classRatioVectorTraining[2]+=1
        if xTrainClass2 is None:
            xTrainClass2 = copyXVector
            xtrainDotClass2 = xDotProduct
        else:
            xTrainClass2 = np.concatenate((xTrainClass2,copyXVector), axis = 0)
            xtrainDotClass2 = np.concatenate((xtrainDotClass2,xDotProduct),axis=0)
            
    if yTrainOutput[z] == 3:
        classRatioVectorTraining[3]+=1
        if xTrainClass3 is None:
            xTrainClass3 = copyXVector
            xtrainDotClass3 = xDotProduct
        else:
            xTrainClass3 = np.concatenate((xTrainClass3,copyXVector), axis = 0)
            xtrainDotClass3 = np.concatenate((xtrainDotClass3,xDotProduct),axis=0)
            
    if yTrainOutput[z] == 4:
        classRatioVectorTraining[4]+=1
        if xTrainClass4 is None:
            xTrainClass4 = copyXVector
            xtrainDotClass4 = xDotProduct
        else:
            xTrainClass4 = np.concatenate((xTrainClass4,copyXVector), axis = 0)
            xtrainDotClass4 = np.concatenate((xtrainDotClass4,xDotProduct),axis=0)
            
    if yTrainOutput[z] == 5:
        classRatioVectorTraining[5]+=1
        if xTrainClass5 is None:
            xTrainClass5 = copyXVector
            xtrainDotClass5 = xDotProduct
        else:
            xTrainClass5 = np.concatenate((xTrainClass5,copyXVector), axis = 0)
            xtrainDotClass5 = np.concatenate((xtrainDotClass5,xDotProduct),axis=0)
            
    if yTrainOutput[z] == 6:
        classRatioVectorTraining[6]+=1
        if xTrainClass6 is None:
            xTrainClass6 = copyXVector
            xtrainDotClass6 = xDotProduct
        else:
            xTrainClass6 = np.concatenate((xTrainClass6,copyXVector), axis = 0)
            xtrainDotClass6 = np.concatenate((xtrainDotClass6,xDotProduct),axis=0)
            
    if yTrainOutput[z] == 7:
        classRatioVectorTraining[7]+=1
        if xTrainClass7 is None:
            xTrainClass7 = copyXVector
            xtrainDotClass7 = xDotProduct
        else:
            xTrainClass7 = np.concatenate((xTrainClass7,copyXVector), axis = 0)
            xtrainDotClass7 = np.concatenate((xtrainDotClass7,xDotProduct),axis=0)
            
    if yTrainOutput[z] == 8:
        classRatioVectorTraining[8]+=1
        if xTrainClass8 is None:
            xTrainClass8 = copyXVector
            xtrainDotClass8 = xDotProduct
        else:
            xTrainClass8 = np.concatenate((xTrainClass8,copyXVector), axis = 0)
            xtrainDotClass8 = np.concatenate((xtrainDotClass8,xDotProduct),axis=0)
            
    if yTrainOutput[z] == 9:
        classRatioVectorTraining[9]+=1
        if xTrainClass9 is None:
            xTrainClass9 = copyXVector
            xtrainDotClass9 = xDotProduct
        else:
            xTrainClass9 = np.concatenate((xTrainClass9,copyXVector), axis = 0)
            xtrainDotClass9 = np.concatenate((xtrainDotClass9,xDotProduct),axis=0)
            
classRatioVectorTraining = classRatioVectorTraining/dataPointsTrain # has P(y=c) for all the classes using training data



## IMPLEMENTING GMM
def gmm(modelCount,xInput,xDotX): # xDotX (dataPints*1)
    ## RUNNING GMM ##
    dataPoints = xInput.shape[0]
    dimensions = xInput.shape[1]
    mixCoeffArray = np.random.rand(modelCount,1)
    tiledMeanMatrix = np.random.rand(dimensions,modelCount)
    covValVector = np.random.rand(modelCount,1)+10 # (modelCount,1)
    
    logSumCurrIt = 0.0
    for i in range(maxIter):
        print("iteration number::", i)
        rMatrix = None
        logSumPrevIt = logSumCurrIt
        for k in range(modelCount):
            meanVectorK = np.reshape(tiledMeanMatrix[:,k],(dimensions,1))#(dimesions*1)
            covValK = covValVector[k]
            mixCoeffK = mixCoeffArray[k]
            uTu = np.dot(np.transpose(meanVectorK),meanVectorK)#(scalar)
            xU = np.dot(xInput,meanVectorK)#(dataPints*1)
            xMeanEuclid = xDotX + uTu -2*xU #(dataPints*1)
            print("zz::", covValK)
            xMeanEuclid = (-1/(2*covValK))*xMeanEuclid
            xMeanEuclid = np.log(mixCoeffK)-(dimensions/2)*np.log(covValK)+xMeanEuclid
            if k == 0:
                rMatrix = xMeanEuclid
            else:
                rMatrix = np.concatenate((rMatrix,xMeanEuclid),axis = 1)
                
        if rMatrix is not None:
            print("Entered here in iteration ", i)
            rMatrix = rMatrix*1./np.max(rMatrix, axis=0)
            rSumVector = np.reshape(np.sum(rMatrix,axis=1),(dataPoints,1))#(dataPoints,1)# will be used for log liklihood again
            rMatrix = rMatrix/rSumVector ## STEP 5 has posterior values
            print("rSumVector negative count::::", (rSumVector<=0).sum())
            logSumCurrIt = -np.sum(rSumVector,axis=0) # scalar
            if i>1 and abs(logSumCurrIt-logSumPrevIt)<=np.abs(logSumCurrIt)*tol:
                break
            vec = np.sum(rMatrix,axis = 0)
            sumPosteriorForKVector = np.reshape(vec, (vec.shape[0],1)) #(modelCount,1)
            
            mixCoeffArray = sumPosteriorForKVector/dataPoints # pi k
            tiledMeanMatrix = None
            for j in range(modelCount):
                postColumnForK = rMatrix[:,j]
                postColumnForK = np.reshape(postColumnForK,(postColumnForK.shape[0],1)) # (dataPoints,1)
                #Evaluate mean vector for this iteration
                meanForKMatrix = postColumnForK*xInput
                meanForKVector = (np.sum(meanForKMatrix,axis=0))/sumPosteriorForKVector[j]
                
                meanForKVector = np.reshape(meanForKVector,(meanForKVector.shape[0],1)) # dimensions*1
                
                if j ==0:
                    tiledMeanMatrix = meanForKVector
                else:
                    tiledMeanMatrix = np.concatenate((tiledMeanMatrix,meanForKVector), axis = 1) # (d*k)after complete loop over modelCount
                
                #Evaluate covValVector
                uTu = np.dot(np.transpose(meanForKVector), meanForKVector) # scalar
                xU = np.dot(xInput,meanForKVector)#(dataPints*1)
                xMeanEuclid = xDotX + uTu -2*xU #(dataPints*1)
                xMeanEuclid = xMeanEuclid*postColumnForK
                covK = (np.sum(xMeanEuclid,axis=0))/(dimensions*sumPosteriorForKVector[j]) # scalar
                covValVector[j] = covK
                
    return mixCoeffArray, tiledMeanMatrix,covValVector


meanMatrixFinal = None
covValVectorFinal = None
mixCoeffVectorFinal = None

print(">>>Starting gmm for class0<<<<")
mixCoeffV0,meanMatrix0,covVal0 = gmm(modelNumber, xTrainClass0, xtrainDotClass0)
meanMatrixFinal = meanMatrix0
covValVectorFinal = covVal0
mixCoeffVectorFinal = mixCoeffV0
print(">>>End gmm for class 0<<<")

print(">>>Starting gmm for class1<<<<")
mixCoeffV1,meanMatrix1,covVal1 = gmm(modelNumber, xTrainClass1, xtrainDotClass1)
meanMatrixFinal = np.concatenate((meanMatrixFinal,meanMatrix1), axis = 1)
covValVectorFinal = np.concatenate((covValVectorFinal,covVal1),axis = 0)
mixCoeffVectorFinal = np.concatenate((mixCoeffVectorFinal,mixCoeffV1),axis = 0)
print(">>>End gmm for class 1<<<")


print(">>>Starting gmm for class2<<<<")
mixCoeffV2,meanMatrix2,covVal2 = gmm(modelNumber, xTrainClass2, xtrainDotClass2)
meanMatrixFinal = np.concatenate((meanMatrixFinal,meanMatrix2), axis = 1)
covValVectorFinal = np.concatenate((covValVectorFinal,covVal2),axis = 0)
mixCoeffVectorFinal = np.concatenate((mixCoeffVectorFinal,mixCoeffV2),axis = 0)
print(">>>End gmm for class 2<<<")


print(">>>Starting gmm for class3<<<<")
mixCoeffV3,meanMatrix3,covVal3 = gmm(modelNumber, xTrainClass3, xtrainDotClass3)
meanMatrixFinal = np.concatenate((meanMatrixFinal,meanMatrix3), axis = 1)
covValVectorFinal = np.concatenate((covValVectorFinal,covVal3),axis = 0)
mixCoeffVectorFinal = np.concatenate((mixCoeffVectorFinal,mixCoeffV3),axis = 0)
print(">>>End gmm for class 3<<<")


print(">>>Starting gmm for class4<<<<")
mixCoeffV4,meanMatrix4,covVal4 = gmm(modelNumber, xTrainClass4, xtrainDotClass4)
meanMatrixFinal = np.concatenate((meanMatrixFinal,meanMatrix4), axis = 1)
covValVectorFinal = np.concatenate((covValVectorFinal,covVal4),axis = 0)
mixCoeffVectorFinal = np.concatenate((mixCoeffVectorFinal,mixCoeffV4),axis = 0)
print(">>>End gmm for class 4<<<")


print(">>>Starting gmm for class5<<<<")
mixCoeffV5,meanMatrix5,covVal5 = gmm(modelNumber, xTrainClass5, xtrainDotClass5)
meanMatrixFinal = np.concatenate((meanMatrixFinal,meanMatrix5), axis = 1)
covValVectorFinal = np.concatenate((covValVectorFinal,covVal5),axis = 0)
mixCoeffVectorFinal = np.concatenate((mixCoeffVectorFinal,mixCoeffV5),axis = 0)
print(">>>End gmm for class 5<<<")


print(">>>Starting gmm for class6<<<<")
mixCoeffV6,meanMatrix6,covVal6 = gmm(modelNumber, xTrainClass6, xtrainDotClass6)
meanMatrixFinal = np.concatenate((meanMatrixFinal,meanMatrix6), axis = 1)
covValVectorFinal = np.concatenate((covValVectorFinal,covVal6),axis = 0)
mixCoeffVectorFinal = np.concatenate((mixCoeffVectorFinal,mixCoeffV6),axis = 0)
print(">>>End gmm for class 6<<<")


print(">>>Starting gmm for class7<<<<")
mixCoeffV7,meanMatrix7,covVal7 = gmm(modelNumber, xTrainClass7, xtrainDotClass7)
meanMatrixFinal = np.concatenate((meanMatrixFinal,meanMatrix7), axis = 1)
covValVectorFinal = np.concatenate((covValVectorFinal,covVal7),axis = 0)
mixCoeffVectorFinal = np.concatenate((mixCoeffVectorFinal,mixCoeffV7),axis = 0)
print(">>>End gmm for class 7<<<")


print(">>>Starting gmm for class8<<<<")
mixCoeffV8,meanMatrix8,covVal8 = gmm(modelNumber, xTrainClass8, xtrainDotClass8)
meanMatrixFinal = np.concatenate((meanMatrixFinal,meanMatrix8), axis = 1)
covValVectorFinal = np.concatenate((covValVectorFinal,covVal8),axis = 0)
mixCoeffVectorFinal = np.concatenate((mixCoeffVectorFinal,mixCoeffV8),axis = 0)
print(">>>End gmm for class 8<<<")


print(">>>Starting gmm for class9<<<<")
mixCoeffV9,meanMatrix9,covVal9 = gmm(modelNumber, xTrainClass9, xtrainDotClass9)
meanMatrixFinal = np.concatenate((meanMatrixFinal,meanMatrix9), axis = 1)
covValVectorFinal = np.concatenate((covValVectorFinal,covVal9),axis = 0)
mixCoeffVectorFinal = np.concatenate((mixCoeffVectorFinal,mixCoeffV9),axis = 0)
print(">>>End gmm for class 9<<<")

## TESTING

##READING DATA FOR TESTING
xTestInput = pd.read_csv('X_test.csv',header=None,nrows=10000).as_matrix()
yTestOutput = pd.read_csv('y_test.csv',header=None,nrows=10000).as_matrix()             

testDataCount = xTestInput.shape[0]
dimensionTest = xTestInput.shape[1]

testDotPrArray = np.zeros((testDataCount,1),dtype = float)
for i in range(testDataCount):
    testDotPrArray[i] = np.dot(xTestInput[i],xTestInput[i])
           
totalGaussians = modelNumber*classNumber

finalProbMatrixTest = None
for k in range(totalGaussians):
    print("Working for Gaussian number", k)
    meanVector = np.reshape((meanMatrixFinal[:,k]),(dimensionTest,1))
    covVal = covValVectorFinal[k]
    uTu = np.dot(np.transpose(meanVector), meanVector)
    Xu = np.dot(xTestInput,meanVector)
    euclidDistance = testDotPrArray + uTu - 2*Xu # (n*1)
    euclidDistance = (-1/(2*covVal))*euclidDistance
    euclidDistance = np.log(mixCoeffVectorFinal[k])-(dimensionTest/2)*np.log(covVal)+euclidDistance
    correspondingClass = int(k/modelNumber)
    euclidDistance = classRatioVectorTraining[correspondingClass]*euclidDistance
    
    if k == 0:
        finalProbMatrixTest = euclidDistance #(n*1)
    else:
        finalProbMatrixTest = np.concatenate((finalProbMatrixTest,euclidDistance),axis = 1)


finalMatrix = np.empty((testDataCount,classNumber)) 
for i in range(classNumber):
    finalMatrix[:,i] = np.sum(finalProbMatrixTest[:,modelNumber*i:modelNumber*(i+1)], axis=1)
        

finalPredictionVector = np.argmax(finalMatrix,axis = 1)

finalPredictionVector = np.reshape((finalPredictionVector),(testDataCount,1)) # predicted class of each data point
errorCount = ((yTestOutput-finalPredictionVector)!=0).sum()
print("Error percent for models :%d is %f" % (modelNumber,(errorCount*100)/testDataCount))

        
    
    
    
    
    
                
            
            
        
            
            
        
    