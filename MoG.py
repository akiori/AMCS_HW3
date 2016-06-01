import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg

class GoM(object):
    def __init__(self, m, n, k):
        self.numOfSample = m
        self.numOfFeature = n
        self.numOfKernel = k
		#mixture coeff:
        self.mixCoef = np.array(([1 / self.numOfKernel] * self.numOfKernel))
        #mean of different kernels
        #self.means = np.random.randint(1, 5, (self.numOfKernel, self.numOfFeature))
        self.means = 10 * np.random.random([self.numOfKernel, self.numOfFeature])
        #Sigmas: Covariance Matrix
        self.covMat = np.zeros([self.numOfKernel, self.numOfFeature, self.numOfFeature], dtype=np.float64)
        
        for i in range(self.numOfKernel):
            self.covMat[i] = np.eye(self.numOfFeature)
		#posterior probability
        self.gamma = np.zeros(shape=[self.numOfSample, self.numOfKernel], dtype=np.float64)

        self.data = np.zeros(([self.numOfSample, self.numOfFeature]), dtype=np.float64)
        #labels
        self.label = np.zeros([self.numOfSample, ], dtype=np.int8)
        #computed labels: λ
        self.lamb = np.zeros([self.numOfSample, ], dtype=np.int8)

	
    def generator(self, mixCoef, means, covMat):
        for i in range(self.numOfSample):
		
            maxId = np.argmax(np.random.multinomial(1, mixCoef, 1))
            self.data[i, :] = np.random.multivariate_normal(means[maxId, :], covMat[maxId, :, :])
            self.label[i] = maxId

    def computeProb(self, x, currMean, currCov):
	
        tmp1 = (2 * np.pi) ** ( 1.0 * self.numOfFeature / 2)
		
        tmp2 = np.sqrt(linalg.det(currCov))
        #print("aaaaa "+str(tmp1))
        #print("bbbbb "+str(tmp2))
        denominator = tmp1 * tmp2
        
        offset = (x - currMean).T #reshape(self.numOfFeature, 1)
        expoItem = float(offset.T.dot(np.linalg.inv(currCov)).dot(offset))
        numerator = np.exp(-1.0 / 2 * expoItem)
        return numerator / denominator
	#given x, compute its posterior probability
    def computePosterior(self, x):
        prob = np.zeros(shape=self.numOfKernel, dtype=np.float64)
        posProb = np.zeros(shape=self.numOfKernel, dtype=np.float64)
        for i in range(self.numOfKernel):
            prob[i] = self.computeProb(x, self.means[i, :], self.covMat[i, :, :])
            posProb[i] = prob[i] * self.mixCoef[i]
        return posProb / posProb.sum()

    def E_step(self):
        for i in range(self.numOfSample):
            self.gamma[i, :] = self.computePosterior(self.data[i, :])

    def M_step(self):
        self.mixCoef = np.mean(self.gamma, 0)
        self.means = self.gamma.T.dot(self.data) / np.sum(self.gamma, 0).reshape(self.numOfKernel, 1)
        for i in range(self.numOfKernel):
            offset = self.data - self.means[i, :]
            numerator = ((self.data - self.means[i, :])*(self.gamma[:, i].reshape(self.numOfSample, 1))).T.dot(offset)
            denominator = np.sum(self.gamma[:, i])
            self.covMat[i, :, :] = numerator / denominator

    def predict(self):
        for i in range(self.numOfSample):
            self.lamb[i] = np.argmax(self.computePosterior(self.data[i, :]))
            #print(self.lamb)

    def validate(self):
        self.predict()
        corrects = np.sum(np.equal(self.lamb, self.label))
        return corrects

    def plot(self):
        #plt.figure(1)
        ax1 = plt.subplot(121)
        ax2 = plt.subplot(122)
        for i in range(self.numOfSample):
            plt.sca(ax1)
            if 0 == self.label[i]:
                plt.plot(self.data[i, 0], self.data[i, 1], 'r^')
            elif 1 == self.label[i]:
                plt.plot(self.data[i, 0], self.data[i, 1], 'yo')
            elif 2 == self.label[i]:
                plt.plot(self.data[i, 0], self.data[i, 1], 'g+')
            else:
                plt.plot(self.data[i, 0], self.data[i, 1], 'b*')

            plt.sca(ax2)
            if 0 == self.lamb[i]:
                plt.plot(self.data[i, 0], self.data[i, 1], 'r^')
            elif 1 == self.lamb[i]:
                plt.plot(self.data[i, 0], self.data[i, 1], 'yo')
            elif 2 == self.lamb[i]:
                plt.plot(self.data[i, 0], self.data[i, 1], 'g+')
            else:
                plt.plot(self.data[i, 0], self.data[i, 1], 'b*')
            
        plt.sca(ax1)
        plt.title('true clusters')
        plt.sca(ax2)
        plt.title('Mixture of Gaussian results')

        plt.show()

if __name__ == '__main__':
    ker = 3
    smp = 100
    ftr = 4

    mix = np.random.randint(1, 20, ker)
	
    _mixCoef = mix / mix.sum()
	
    #print(_mixCoef)
    _means = 10 * np.random.random((ker, ftr))
    _covMat = np.zeros(shape=(ker, ftr, ftr), dtype=np.float64)
    for i in range(ker):
        tmp = np.eye(ftr) * np.random.randint(i + 1, i * i + 2)
        _covMat[i] = tmp

    gmm_object = GoM(smp, ftr, ker)
    gmm_object.generator(_mixCoef, _means, _covMat)

    max_iter = 1000
    for it in range(max_iter):
        #print(it)
        gmm_object.E_step()
        gmm_object.M_step()
        
	#print(GoM_object.lamb)
    precision = gmm_object.validate() / smp
    print('precision is ', precision)
    gmm_object.plot()