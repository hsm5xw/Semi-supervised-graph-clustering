class CrossValidation:
    def Estimate(sigmaGrid, data, numFolds, k, classRanges, numIt, labels):
        n = data.shape[0]
        simMatrices = []

        for i in range(numFolds):
            numb_cv = n/numFolds
            ind_all = np.linspace(0, n - 1, n)
            ind_all = ind_all.astype(int)

            idx_test = ind_all[k * numb_cv:(k + 1) * numb_cv]
            idx_train = np.delete(ind_all, idx_test)

            Xtr_CV = data[idx_train,:]
            ytr_CV = labels[idx_train]
