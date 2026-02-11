
from datetime import datetime
from dateutil.relativedelta import relativedelta as rd
from time import strptime

class fitthb:
	def __init__(self, q = 0.0, bf = 0.0,di = 0.0,t = 0.0,buildupMonths = 0.0,buildupRate = 0.0,eur = 0.0,dMin = 0.0,bi = 2.0,fcMos = 600, phase = "", id = "", startDate="", _offset=0, _cumul= 0.0):
		print("initializing")
		self.q = q
		self.bi = bi
		self.bf = bf
		self.di = di
		self.t = t
		self.buildupMonths = buildupMonths
		self.buildupRate = buildupRate
		self.eur = eur
		self.dMin = dMin
		self.fcMos = fcMos
		self.phase = phase
		self.id = id
		self.startDate = startDate
		self._offset = _offset
		self._cumul = _cumul
	def __str__(self):
		return str(self.__dict__)

	def setFCStartDate(self,firstProdDate, adjMos):
		#startDate = datetime.date(1900,1,31)

		try:
			startDate = datetime.strptime(firstProdDate, "%m/%d/%Y")
			startDate = startDate + rd(months = adjMos)
		except:
			print("Could not parse " + firstProdDate + " for well " + self.id)
		self.startDate = str(startDate)

	def setFCOffset(self,prodMonths):
		self._offset = prodMonths

	def setCumul(self,cumul):
		self._cumul = cumul

	def setCumulFromRate(self,rates):
		cumul = 0.0
		for i in range(0,len(rates) - 1):
			cumul += rates[i] * 30.4
		self._cumul = cumul
	def getCumul(self):
		return self._cumul

if __name__ == '__main__':

	thb = fitthb(q = 1807.77, bf = 1.14,di = 0.49,t = 0.0,buildupMonths = 25,buildupRate = 964.15,eur = 2872741.59,dMin = 0.05,bi = 2.0,fcMos = 600, phase = "Gas", id = "238", startDate="2/29/2012", _offset=0, _cumul= 0.0)

	#thb.startDate = "12/28/2011"
	#print(thb.startDate)
	#thb.setFCStartDate(thb.startDate,2)
	print(thb)
