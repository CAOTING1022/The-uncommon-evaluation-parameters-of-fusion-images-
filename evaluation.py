import math
import numpy as np
import cv2
from skimage import measure,metrics
from scipy import signal
#求图像的灰度概率分布
def gray_possiblity(image):
    tmp = np.zeros(256,dtype='float')
    val = 0
    k = 0
    res = 0
    img = np.array(image)
    for i in range(len(img)):
        for j in range(len(img[i])):
            val = img[i][j]             #像素值
            val = val.astype(int)
            tmp[val] = float(tmp[val] + 1) #该像素值的次数
            k = k+1              #总次数

    for i in range(len(tmp)):
        tmp[i] = float(tmp[i] / k)   #各个灰度概率
    return tmp

#求图像的条件熵： H(x|y) = H(x,y) - H(y)
def condition_entropy(image,fuse):
    tmp = np.zeros((256,256),dtype='float')
    res = 0
    k = 0
    image = np.array(image,dtype='int')
    fuse = np.array(fuse,dtype='int')
    rows,cols = image.shape[:2]
    for i in range(len(image)):
        for j in range(len(image[i])):
            tmp[image[i][j]][fuse[i][j]] = float(tmp[image[i][j]][fuse[i][j]]+1)
            k = k+1
    for i in range(256):
        for j in range(256):
            p = tmp[i,j]/k
            if p!=0:
                res = float(res-p*math.log(p,2))
    return res - Entropy(fuse)

#求图像的信息熵 P(a)表示灰度概念 entropy = PA(a)*logPA(a)
def Entropy(image):
    tmp = []
    res = 0
    tmp = gray_possiblity(image)
    for i in range(len(tmp)):
        if(tmp[i]==0):
            res = res
        else:
            res = float(res - tmp[i]*(math.log(tmp[i],2)))
    return res

#交叉熵 求和 -(p(x)logq(x)+(1-p(x))log(1-q(x)))
def cross_entropy(image,aim):
    tmp1 = []
    tmp2 = []
    res = 0
    tmp1 = gray_possiblity(image)
    tmp2 = gray_possiblity(aim)
    for i in range(len(tmp1)):
        if(tmp1[i]!=0)and(tmp2[i]!=0):
            res = tmp1[i]*math.log(1/tmp2[i]) + res
            #res = float(res-(tmp1[i]*math.log2(tmp2[i])+(1-tmp1[i])*math.log2(1-tmp2[i])))
    return res

#求psnr 利用skimage库
def peak_signal_to_noise(true,test):
    return metrics.peak_signal_noise_ratio(true,test,data_range=255)

#求IQM 论文参考文献14
def IQM(image,fused):
    image = np.array(image,float)
    fused = np.array(fused,float)
    N = len(image)*len(image[0])
    k = 0.0
    x_mean = 0.0
    y_mean = 0.0
    for i in range(len(image)):
        for j in range(len(image[i])):
            x_mean = x_mean + image[i][j]
            y_mean = y_mean + fused[i][j]
            k = k+1
    x_mean = float(x_mean/k)
    y_mean = float(y_mean/k)
    for i in range(len(image)):
        for j in range(len(image[i])):
            xy = xy+(image[i][j]-x_mean)*(fused[i][j]-y_mean)
            x = x+(image[i][j]-x_mean)*(image[i][j]-x_mean)
            y = y+(fused[i][j]-y_mean)*(fused[i][j]-y_mean)
    xy = xy/(N-1)
    x = x/(N-1)
    y = y/(N-1)
    Q = 4*xy*x_mean*y_mean/((x*x+y*y)*(x_mean*x_mean+y_mean*y_mean))
    return Q


#计算图像的Qabf
def matrix_pow(m):
    for i in range(len(m)):
        for j in range(len(m[i])):
            m[i][j] = pow(m[i][j],2)
    return m
def matrix_sqrt(m):
    for i in range(len(m)):
        for j in range(len(m[i])):
            m[i][j] = math.sqrt(m[i][j])
    return m
def matrix_multi(m1,m2):
    m = np.zeros(m1.shape)
    for i in range(len(m1)):
        for j in range(len(m1[i])):
            m[i][j] = m1[i][j]*m2[i][j]
    return m
def Qabf(pA,pB,pF):
    L = 1
    Tg = 0.9994
    Kg = -15
    Dg = 0.5
    Ta = 0.9879
    Ka = -22
    Da = 0.8
    h3 = [[-1,0,1],
          [-2,0,2],
          [-1,0,1]]
    h3 = np.array(h3)
    h3 = h3.astype(np.float)
    h1 = [[1,2,1],
          [0,0,0],
          [-1,-2,-1]]
    h1 = np.array(h1)
    h1 = h1.astype(np.float)
    h2 = [[0,1,2],
          [-1,0,1],
          [-2,-1,0]]
    h2 = np.array(h2)
    h2 = h2.astype(np.float)
    SAx  = signal.convolve2d(pA,h3,'same')
    SAy = signal.convolve2d(pA,h1,'same')
    gA = matrix_pow(SAx)+matrix_pow(SAy)
    gA = matrix_sqrt(gA)
    M = SAy.shape[0]
    N = SAy.shape[1]
    aA = np.zeros(SAx.shape)
    for i in range(M):
        for j in range(N):
            if SAx[i][j]==0:
                aA[i][j] = math.pi/2
            else:
                aA[i][j] = math.atan(SAy[i][j]/SAx[i][j])

    SBx = signal.convolve2d(pB,h3,'same')
    SBy = signal.convolve2d(pB,h1,'same')
    gB = matrix_pow(SBx) + matrix_pow(SBy)
    gB = matrix_sqrt(gB)
    aB = np.zeros(SBx.shape)
    for i in range(SBx.shape[0]):
        for j in range(SBx.shape[1]):
            if SBx[i][j] == 0:
                aB[i][j] = math.pi/2
            else:

                aB[i][j] = math.atan(SBy[i][j]/SBx[i][j])
    SFx = signal.convolve2d(pF,h3,boundary='symm',mode='same')
    SFy = signal.convolve2d(pF,h1,boundary='symm',mode='same')
    gF = matrix_sqrt((matrix_pow(SFx)+matrix_pow(SFy)))
    M = SFx.shape[0]
    N = SFx.shape[1]
    aF = np.zeros(SFx.shape)
    for i in range(M):
        for j in range(N):
            if SFx[i][j] == 0:
                aF[i][j] = math.pi/2
            else:
                aF[i][j] = math.atan(SFy[i][j]/SFx[i][j])
    #the relative strength and orientation value of GAF,GBF and AAF,ABF
    GAF = np.zeros(SFx.shape)
    AAF = np.zeros(SFx.shape)
    QgAF = np.zeros(SFx.shape)
    QaAF = np.zeros(SFx.shape)
    QAF  = np.zeros(SFx.shape)
    for i in range(M):
        for j in range(N):
            if gA[i][j]>gF[i][j]:
                GAF[i][j] = gF[i][j]/gA[i][j]
            else:
                if gF[i][j] == 0:
                    GAF[i][j] = 0
                else:
                    GAF[i][j] = gA[i][j] / gF[i][j]

            AAF[i][j] = 1- abs(aA[i][j]-aF[i][j])/(math.pi/2)
            QgAF[i][j] = Tg / (1+math.exp(Kg*(GAF[i][j]-Dg)))
            QaAF[i][j] = Ta / (1+math.exp(Ka*(AAF[i][j]-Da)))
            QAF[i][j] = QgAF[i][j]*QaAF[i][j]

    GBF = np.zeros(SFx.shape)
    ABF = np.zeros(SFx.shape)
    QgBF = np.zeros(SFx.shape)
    QaBF = np.zeros(SFx.shape)
    QBF = np.zeros(SFx.shape)
    for i in range(M):
        for j in range(N):
            if gB[i][j] == gF[i][j]:
                GBF[i][j] = gF[i][j]
            else:
                if gF[i][j]==0:
                    GBF[i][j] = 0
                else:
                    GBF[i][j] = gB[i][j]/gF[i][j]
            ABF[i][j] = 1 - abs(aB[i][j]-aF[i][j])/(math.pi/2)
            QgBF[i][j] = Tg / (1 + math.exp(Kg*(GBF[i][j]-Dg)))
            QaBF[i][j] = Ta / (1 + math.exp(Ka*(ABF[i][j]-Da)))
            QBF[i][j] = QgBF[i][j]*QaBF[i][j]
    # compute the QABF
    deno = np.sum(np.sum(gA+gB))
    nume = np.sum(np.sum(matrix_multi(QAF,gA)+matrix_multi(QBF,gB)))
    output = nume/deno
    return  output

#测试集
if __name__ == '__main__':
    c = np.array([[1, 2, 3],
                  [-1, 1, 4]])

    print(np.sum(np.linalg.norm(c, ord=1, axis=0)))
    img = [[1, 2, 3, 4],
           [5, 6, 7, 9]]
    print(len(img))
    print(len(img[0]))
    image = cv2.imread('001.jpg', 0)
    image1 = cv2.imread('001.jpg', 0)
    print(image.shape)
    print(Entropy(image))
    print(cross_entropy(image, image1))
    print(Qabf(image,image1,image))
    print(measure.shannon_entropy(image, base=2))
