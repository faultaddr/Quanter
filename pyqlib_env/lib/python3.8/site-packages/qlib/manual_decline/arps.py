import math as math
from itertools import chain

def generateVolumes(qi,bi,di,nMos,dMin):
    # Init
    isExp = False;
    tExp = 0;
    qiExp = qi;
    r = {"rates":[],"cums":[]}

    # QC
    if( bi == 1.0 ):
        bi = 0.999
    if( bi == 0.0 ):
        bi = 0.001

    # Start Loop
    qStart = []
    qEnd = []
    dNom = []
    dEff = []
    cums = []
    rates = []
    dnomLim = -math.log(1.0 - dMin) / 12.0
    dnom = ((math.pow((1.0 - di), (-bi)) - 1.0) / bi)
    # Buildup
    if( di == 0.0 ):
        for i in range(0,nMos):
            rates.append(qi)
            cums.append(qi * 30.4167)
        #r["rates"] = [x*30.4167 for x in rates]
        #r["cums"] = cums
            yield qi * 30.4167
        return
    # Decline
    for i in range(0,nMos):
        # Start and ending rates
        t = i + 1;
        if( i==0 ):
            qStart.append(qi)
        else:
            qStart.append(qEnd[i-1])
        dNom.append(dnom * math.pow(qStart[i] / qi,bi))
        dEff.append(1.0 - math.exp(-dNom[i]))
        qEnd.append(qi / math.pow( (1.0 + bi * dnom / 12.0 * t), (1.0 / bi)))

        # Cumulative production
        if( dEff[i] > dMin ):
            cums.append(30.4167 * (qi / ( (dnom / 12.0) *(bi - 1.0)))*(( math.pow((1.0 + bi * (dnom / 12.0) * t ),(1.0 - (1.0 / bi))) ) - 1.0))
        else:
            if( not(isExp) ):
                # If this is the first exponential, then create a reference
                isExp = True
                tExp = t - 1
                if( i > 0 ):
                    qiExp = qEnd[i-1];
            cums.append(30.4167 * (qiExp / dnomLim)*(1.0 - math.exp(-dnomLim * (t - tExp))) + cums[max(tExp - 1,0)])
        eur = cums[i]

        # Rates
        if( i == 0 ):
            rates.append(cums[i] / 30.4167)
        else:
            rates.append((cums[i] - cums[i - 1]) / 30.4167)
    #r["rates"] = [x*30.4167 for x in rates]
    #r["cums"] = cums
    #return r
        yield rates[-1] * 30.4167

def calculateRateValue(q, b, d, t):
	# Calculate nominal decline
	dNom = (math.pow(1.0 - d, -b) - 1.0) / b / 12.0

	# Estimate rate snapshot
	qn = q / math.pow( (1.0 + b * dNom * t), (1.0 / b))

	return qn
