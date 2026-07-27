import numpy as np
from loadingSolidDynamicalData import bringDynamicalLoadDispData as bb
from loadingDynamicalData import bringDynamicalLoadDispData as aa


def importData(names, remove_dts, remove_noise=True, return_list=False, remove_corners = False,
              cutoff_incr = [699]):
    assert len(remove_dts) == len(names)

    trainitersolidDispData = []
    trainiterLoadData = []
    trainRepsolDispDataConv = []
    chosenIters = []
    size_forSolid = []
    size_forFluid = []

    for i in range(len(names)):
        _, _, _, trainitersolidDispData_, trainiterLoadData_, trainRepsolDispDataConv_, \
                    _,_, _ = bb(cutoff_names= [names[i]], cutoff_incr = cutoff_incr,
                                remove_corners=remove_corners)
    
        if remove_dts[i] is not None:
            remove_noisy = np.load(names[i]+"iters.npy")[:remove_dts[i]].sum()

            if remove_noisy:
                remove_noisy -= remove_dts[i]
                chosenIters = np.concatenate((np.array([0]),
                            np.cumsum(
                                np.load(names[i]+"iters.npy")[1:cutoff_incr[0]]
                            )))
    
            trainitersolidDispData_ = np.delete(trainitersolidDispData_, chosenIters, 1)[:, remove_noisy:]
            trainiterLoadData_ = np.delete(trainiterLoadData_, chosenIters, 1)[:, remove_noisy:]
            trainRepsolDispDataConv_ = np.delete(trainRepsolDispDataConv_, chosenIters, 1)[:, remove_noisy:]
    
        trainitersolidDispData.append(trainitersolidDispData_.copy())
        trainiterLoadData.append(trainiterLoadData_.copy())
        trainRepsolDispDataConv.append(trainRepsolDispDataConv_.copy())
        size_forSolid.append(trainitersolidDispData_.shape[1])
    
    loadData_forSolid = np.hstack(trainiterLoadData)
    dispConvData_forSolid = np.hstack(trainRepsolDispDataConv)
    dispData_forSolid = np.hstack(trainitersolidDispData)


    trainiterLoadData2 = []
    trainiterDispData = []
    trainRepflLoadDataConv = []
    chosenIters = []
    
    for i in range(len(names)):
        _, _, _, trainiterLoadData_, trainiterDispData_, trainRepflLoadDataConv_, \
                    _,_, _ = aa(cutoff_names = [names[i]], cutoff_incr = cutoff_incr,
                                remove_corners=remove_corners)
        # =====  At this point, the iterations of the first time step are removed
    
        if remove_dts[i] is not None:
            remove_noisy = np.load(names[i]+"iters.npy")[:remove_dts[i]].sum()
    
            if remove_noisy:
                remove_noisy -= remove_dts[i]
                chosenIters = np.concatenate((np.array([0]),
                            np.cumsum(
                                np.load(names[i]+"iters.npy")[1:cutoff_incr[0]]
                            )))
    
            trainiterLoadData_ = np.delete(trainiterLoadData_, chosenIters, 1)[:, remove_noisy:]
            trainiterDispData_ = np.delete(trainiterDispData_, chosenIters, 1)[:, remove_noisy:]
            trainRepflLoadDataConv_ = np.delete(trainRepflLoadDataConv_, chosenIters, 1)[:, remove_noisy:]
    
        trainiterLoadData2.append(trainiterLoadData_.copy())
        trainiterDispData.append(trainiterDispData_.copy())
        trainRepflLoadDataConv.append(trainRepflLoadDataConv_.copy())
        size_forFluid.append(trainiterLoadData_.shape[1])

    loadData_forFluid = np.hstack(trainiterLoadData2)
    dispData_forFluid = np.hstack(trainiterDispData)
    loadConvData_forFluid = np.hstack(trainRepflLoadDataConv)


    if not return_list:
        return loadData_forFluid, dispData_forFluid, loadConvData_forFluid, size_forFluid, loadData_forSolid, dispData_forSolid, dispConvData_forSolid, size_forSolid
    else:
        return trainiterLoadData2, trainiterDispData, trainRepflLoadDataConv, size_forFluid, trainiterLoadData, trainitersolidDispData, trainRepsolDispDataConv, size_forSolid

