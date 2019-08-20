def groupParams(m,lr_factors = [1]):
    params_arr = getParamsArr(m)
    paramsGroupRanges = [ [i, min( i + len(params_arr)// len(lr_factors), len(params_arr))]
                         for i in range(0, len(params_arr),len( params_arr)//len(lr_factors))]
    paramsGroups = [ params_arr[ p[0] : p[1]] for p in paramsGroupRanges]
    return paramsGroups

def getParamsArr(m):
    if (list(m.children())):
        childParams = sum([ getParamsArr(c) for c in m.children()],[])
        loneParams = [p for p in m.parameters() if id(p) not in [id(cp) for cp in childParams]]
        return childParams + loneParams
    else:
        return list(m.parameters())
