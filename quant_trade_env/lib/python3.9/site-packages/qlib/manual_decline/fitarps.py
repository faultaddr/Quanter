import qlib.manual_decline
from qlib.manual_decline.arpsfcn import arpsfcn as arpsfcn
import math as math
class fitarps:
	#dMin = None

	def __init__(self,buildupMonths = 0, buildupRate = 0,q1 = 0,b1=0,d1=0,t1=0,q2=0,b2=0,d2=0,t2=0,dMin = 0.05,phase="",id="",_startDate = None,_refDate = None, _offset = 0, _cumul = 0.0 ):
		print("initializing")
		self.buildupMonths = buildupMonths
		self.buildupRate = buildupRate
		self.q1 = q1
		self.b1 = b1
		self.d1 = d1
		self.t1 = t1
		self.q2 = q2
		self.b2 = b2
		self.d2 = d2
		self.t2 = t2
		self.dMin = dMin
		self._startDate = _startDate
		self._refDate = _refDate
		self.phase = phase
		self.id = id
		self._offset = _offset
		self._cumul = _cumul

	def __str__(self):
		return str(self.__dict__)


	def getRates(self, t):
		print("DMIN: ",self.dMin)
		#leg 0
		arps = arpsfcn()
		arps.generateRates(self.buildupRate, 0.0,0.0,0.0, self.buildupMonths)
		eur = arps.getDouble("eur")
		rates0 = arps.getArray("rates")
		#leg 1
		arps = arpsfcn()
		arps.generateRates(self.q1, self.b1, self.d1, self.dMin, int(self.t1))
		eur += arps.getDouble("eur")
		rates1 = arps.getArray("rates")

		#leg 2
		arps = arpsfcn()
		arps.generateRates(self.q2,self.b2,self.d2,self.dMin, min(t - int(self.t1) - int(self.buildupMonths), int(self.t2)))
		eur += arps.getDouble("eur")
		rates2 = arps.getArray("rates")

		#stitch the arrays together
		rates = rates0 + rates1 + rates2
		return rates

	def getEUR(self,t):
		#init
		rates = self.getRates(t)

		return sum(rates) * 30.4167

	def getEURWithOffsetVariableFrame(self,t):
		#init
		cumul = 0.0

		#calculate the rates
		rates = self.getRates(t + self._offset)
		nMos = len(rates)
		cums = [0]

		#Accumulate the entire stream
		for i in range(self._offset, nMos - 1):
			cums.insert(i,cums[i] + rates[i] * 30.4167)
			cumul += cums[i]
		return cumul
	def setStartDate(self,startDate):
		self._startDate = startDate

	def getCumul(self):
		return self._cumul



if __name__ == '__main__':
	fitarps = fitarps(buildupMonths = 25, buildupRate = 679.54,q1 = 986.04,b1=2.00,d1=0.5709930255535098,t1=0.0,q2=986.04,b2=1.26,d2=0.64604,t2=575.0,dMin = 0.05,phase="Gas",id="238",_startDate = None,_refDate = None)
	print("Rates: ",fitarps.getRates(600))
	print("EUR: ",fitarps.getEUR(600))