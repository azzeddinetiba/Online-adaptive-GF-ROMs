import numpy as np

def bringDynamicalLoadDispData(cutoff_names = ["fomData/1.5/coSimData/"], cutoff_incr = [699], remove_corners=False):

    assert len(cutoff_names) == len(cutoff_incr)

    RepflLoadDataConv = []
    iterLoadData = []
    iterDispData = []

    trainRepflLoadDataConv = []
    trainiterLoadData = []
    trainiterDispData = []

    testRepflLoadDataConv = []
    testiterLoadData = []
    testiterDispData = []

    cutoffs = {}
    for i in range(len(cutoff_names)):
        cutoffs[cutoff_names[i]] =  cutoff_incr[i]

    map = np.load(cutoff_names[0]+"map_used.npy")

    for i in cutoff_names:
        iters = np.load(i+"/iters.npy")
        if remove_corners:
            flLoadData = np.load(i+"/fluid_load_data_from_fluid.npy")[2:-2, :]
        else:
            flLoadData = np.load(i+"/fluid_load_data_from_fluid.npy")
        dispData = np.load(i+"/disp_data.npy")[map, :]#[:, :iters[:449].sum()]
        dispData = np.vstack((dispData, np.load(i+"/velocity_data.npy")[map, :]))#[:, :iters[:449].sum()]
        lastIters = iters.cumsum()-1

        trainLastId = lastIters[cutoffs[i]]

        flLoadDataConv = flLoadData[:, lastIters][:, :-1]
        flLoadDataConv_rep = np.repeat(flLoadDataConv, iters[1:], axis = 1)
        iterLoadData_ = flLoadData[:, iters[0]:]
        iterDispData_ = dispData[:, iters[0]:]
        RepflLoadDataConv.append(flLoadDataConv_rep.copy())
        iterLoadData.append(iterLoadData_.copy())
        iterDispData.append(iterDispData_.copy())


        trainflLoadDataConv_rep = flLoadDataConv_rep[:, :trainLastId]
        trainiterLoadData_ = iterLoadData_[:, :trainLastId]
        trainiterDispData_ = iterDispData_[:, :trainLastId]
        trainRepflLoadDataConv.append(trainflLoadDataConv_rep.copy())
        trainiterLoadData.append(trainiterLoadData_.copy())
        trainiterDispData.append(trainiterDispData_.copy())

        testflLoadDataConv_rep = flLoadDataConv_rep[:, trainLastId:]
        testiterLoadData_ = iterLoadData_[:, trainLastId:]
        testiterDispData_ = iterDispData_[:, trainLastId:]
        testRepflLoadDataConv.append(testflLoadDataConv_rep.copy())
        testiterLoadData.append(testiterLoadData_.copy())
        testiterDispData.append(testiterDispData_.copy())

    RepflLoadDataConv = np.hstack((RepflLoadDataConv))
    iterLoadData = np.hstack((iterLoadData))
    iterDispData = np.hstack((iterDispData))

    trainRepflLoadDataConv = np.hstack((trainRepflLoadDataConv))
    trainiterLoadData = np.hstack((trainiterLoadData))
    trainiterDispData = np.hstack((trainiterDispData))

    testRepflLoadDataConv = np.hstack((testRepflLoadDataConv))
    testiterLoadData = np.hstack((testiterLoadData))
    testiterDispData = np.hstack((testiterDispData))

    return iterLoadData, iterDispData, RepflLoadDataConv, trainiterLoadData, trainiterDispData, trainRepflLoadDataConv, testiterLoadData, testiterDispData, testRepflLoadDataConv
