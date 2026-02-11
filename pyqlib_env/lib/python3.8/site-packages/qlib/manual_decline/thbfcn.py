import math as math
class thbfcn:

	#class variables
	rates = []
	bAvg = []
	cumul = 0.0
	eurProxy = 0.0
	def __init__(self, rates = [], bAvg = [], cumul=0.0, eurProxy=0.0):
		self.rates = rates
		self.bAvg = bAvg
		self.cumul = cumul
		self.eurProxy = eurProxy
	def __str__(self):
		return str(self.__dict__)
	def generateRatesTHB(self,qi=0,bi=2,bf=0,di=0,telf=0,dMin=0,timeData=[]):
		#check telf domain
		if(telf <= 0):
			telf = 0.001
		#get declines
		dnom = ((math.pow((1.0 - di), (-2.0)) - 1.0) / 2.0) / 365.0
		#get s1 base
		s1_base = 1 / dnom
		#Setup the arrays
		nPts = len(timeData)
		bfArray = []
		bAvg = []
		cumD = []
		d = []
		dbase = []
		cumProd =[]
		calcRates = []
		dTime = []
		dnomLim = (math.pow(1.0 - dMin, -bf) - 1.0) / bf / 365.0
		#print(timeData)
		#Enter the rate calc
		for i in range(0,nPts):
			#start the b calcs
			bfArray.insert(i,bi - (bi - bf) * math.exp(-math.exp(-(math.exp(0.57722) / (1.5 * telf)) * (timeData[i] - telf) + math.exp(0.57722))))
			if (i == 0):
				cumD.insert(i,s1_base)
				d.insert(i, 1.0 / cumD[i])
				dbase.insert(i,1.0)
				#start the rate calcs
				cumProd.insert(i,dbase[0])
				calcRates.insert(i,qi)
				cumul = calcRates[i] * 30.4

		#get declines
		dnom = ((math.pow((1.0 - di), (-2.0)) - 1.0) / 2.0) / 365.0
		#get s1 base
		s1_base = 1 / dnom
		#Setup the arrays
		nPts = len(timeData)
		bfArray = []
		bAvg = []
		cumD = []
		d = []
		dbase = []
		cumProd =[]
		calcRates = []
		dTime = []
		dnomLim = (math.pow(1.0 - dMin, -bf) - 1.0) / bf / 365.0
		#Enter the rate calc
		for i in range(0,nPts):
			#start the b calcs

			bfArray.insert(i,bi - (bi - bf) * math.exp(-math.exp(-(math.exp(0.57722) / (1.5 * telf)) * (timeData[i] - telf) + math.exp(0.57722))))
			if (i == 0):
				cumD.insert(i,s1_base)
				d.insert(i, 1.0 / cumD[i])
				dbase.insert(i,1.0)
				#start the rate calcs
				cumProd.insert(i,dbase[0])
				calcRates.insert(i,qi)
				cumul = calcRates[i] * 30.4

				continue
			else:
				#calc the time delta
				dTime.insert(i - 1,timeData[i] - timeData[i - 1])
				#calc the average b
				bAvg.insert(i - 1,(bfArray[i] - bfArray[i - 1]) / 2 + bfArray[i - 1])
				#calc the decline
				cumD.insert(i, bAvg[i - 1] * dTime[i - 1] + cumD[i - 1])
				d.insert(i,1 / cumD[i])
				d.insert(i,max(d[i],dnomLim))
				dD = d[i] - d[i - 1]
				dbase.insert(i,math.exp(-dTime[i - 1] * (dD / 2 + d[i - 1])))
				#calc the rate data
				cumProd.insert(i, dbase[i] * cumProd[i - 1])
				calcRates.insert(i, cumProd[i] * qi)
				cumul = cumul + calcRates[i] * 30.4
		#print(dTime)
		#Arp's EUR approximation
		qx = calcRates[nPts - 1]
		qab = 16.0
		dx = d[nPts - 1] * 365 / 12
		bx = bfArray[nPts - 1]
		dlx = 0.05
		v1 = math.pow(qx,bx) / ((1.0 - bx) * dx * 12.0)
		dLimNom = (math.pow((1.0-dlx), -bx) -1.0) / bx * (1 / 12.0)
		qLim = qx * math.pow((dLimNom / dx), 1.0 / bx)
		dQ = (qLim - qab) / dlx * 365.0
		v2 = (math.pow(qx, 1.0 - bx)) - math.pow(qLim, 1.0 - bx)
		#setup return class
		self.bAvg = bAvg
		self.rates= calcRates
		self.cumul = cumul
		self.eurProxy = v1 * v2 * 365.0 + dQ + cumul
		#return v1 * v2 * 365.0 + dQ + cumul


	#getter functions
	def getArray(self,v):
		value = v.lower()
		return {"rates": self.rates,
				"bAvg": self.bAvg}[value]
	def getDouble(self,v):
		value = v.lower()
		return {"cumul": self.cumul,
				"eurprox": self.eurProxy}[value]


if __name__ == '__main__':
	qi = 679.54
	di = 0.569999999
	bf = 1.26
	telf = 0.0
	dMin = 0.05
	fcmons = 600
	timedata = []
	for i in range(0,fcmons):
		timedata.insert(i,(30.4167 * i))
	thb = thbfcn([],[],0.0,0.0)
	#print(timedata)
	thb.generateRatesTHB(qi,2,bf,di,telf,dMin,timedata)
	print(thb)
	#thb.startDate = "12/28/2011"
	#print(thb.startDate)
	#thb.setFCStartDate(thb.startDate,2)
	print(thb)
