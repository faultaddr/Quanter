import math as math
# from brennan usage module directly on local Q Docker - PIP wont need these import
#from . import thb2arps as thb2arps
#from . import fitthb as fitthb
#from . import fitarps as fitarps
class arpsfcn:
	rates = []
	cums = []
	#eur = 0.0
	def __init__(self,eur=0.0):
		self.eur = eur


	def generateRates(self,qi, bi, di, dMin, nMos):
		#print("arpsfcn DMIN: ",dMin)
		isExp = False
		tExp = 0
		qiExp = qi
		if(bi == 1.0):
			bi = 0.999
		if(bi == 0.0):
			bi = 0.001

		#startLoop
		qStart = []
		qEnd = []
		dNom = []
		dEff = []
		self.cums = [0]
		self.rates = []
		dnomLim = -math.log(1.0 - dMin) / 12.0
		dnom = ((math.pow((1.0 - di), (-bi)) - 1.0) / bi)

		#buildup
		if(di == 0.0):
			for i in range(0,nMos):
				#print(self.cums)
				self.rates.insert(i,qi)
				if (len(self.cums) == 0):
					newval = qi * 30.4167
				else:
					#print(self.cums,i)
					newval = self.cums[i] + qi * 30.4167
				self.cums.insert(i,newval)
			return
		#decline
		for i in range(0, nMos):
			#start and ending rates
			t = i + 1
			if (i == 0):
				qStart.insert(i,qi)
				#qStart[i] = qi
			else:
				qStart.insert(i,qEnd[i-1])
				#qStart[i] = qEnd[i - 1]
			dNom.insert(i,dnom * math.pow(qStart[i] / qi, bi))
			dEff.insert(i,1.0 - math.exp(-dNom[i]))
			#dEff[i] = 1.0 - math.exp(-dNom[i])
			qEnd.insert(i,qi / math.pow( (1.0 + bi * dnom / 12.0 * t), (1.0 / bi)))
			#qEnd[i] = qi / math.pow( (1.0 + bi * dnom / 12.0 * t), (1.0 / bi))

			#cumulative Production
			if (dEff[i] > dMin):
				self.cums.insert(i,30.4167 * (qi / ( (dnom / 12.0) *(bi - 1.0)))*(( math.pow((1.0 + bi * (dnom / 12.0) * float(t) ),(1.0 - (1.0 / bi))) ) - 1.0))

				#this.cums[i] = 30.4167 * (qi / ( (dnom / 12.0) *(bi - 1.0)))*(( Math.pow((1.0 + bi * (dnom / 12.0) * float(t) ),(1.0 - (1.0 / bi))) ) - 1.0)
			else:
				if( isExp == False):
					isExp = True
					tExp = t - 1
					if (i > 0):
						qiExp = qEnd[i - 1]
				self.cums.insert(i,30.4167 * (qiExp / dnomLim) * (1.0 - math.exp(-dnomLim * (t - tExp))) + self.cums[max(tExp - 1, 0)])
			self.eur = self.cums[i]


			#rates
			if( i == 0):
				self.rates.insert(i,self.cums[i] / 30.4167)
			else:
				self.rates.insert(i,(self.cums[i] - self.cums[i - 1]) / 30.4167)
		#return self.rates # added by David for Qlib Testing
	#getter functions
	def getArray(self,v):
		value = v.lower()
		return {"rates": self.rates,
				"cums": self.cums}[value]

	def getDouble(self,v):
		value = v.lower()
		return {"eur": self.eur}[value]



def lambda_handler(event, context):
	print("EVENT: ",event)
	#if local test
	if (type(event['body']) == dict):
		 data = event['body']['data']
		 #Else if coming from API gateway
	else:
		 data = json.loads(event['body'])
		 data = data['data']
	print("DATA: ", data["data"])
	data = data["data"]
	dMin = data["dMin"][0]
	bi = data["bi"][0]
	nMos = data["buildupMonths"][0]
	qi = data["ip"][0]
	di = data["di"][0]
	apikey = event['body']["apiKey"]
	bf = data["bf"][0]
	fcMos = data["fcMos"]
	t = data["t"][0]
	buildupRate = data["buildupRate"][0]
	#owner = data["ProjectOwner"]

	# fcn = arpsfcn()
	# rate = fcn.generateRates(qi, bi, di, dMin, nMos)
	# eur = fcn.getDouble("eur")
	# print("arpseur: ",eur)
	# res = fitthb.fitthb(qi,bf,di,t,nMos,buildupRate,eur,dMin,bi,fcMos)
	#
	#  #conn_data = get_engine_string(apikey)
	#  #res = fitthb(seg_name, segdata, user, owner, projname, descrip, inputs, conn_data)
	# print("res: ", res)
	converter = thb2arps.thb2arps()
	#initialize thb object
	thb = fitthb.fitthb(qi, bf,di,t,nMos,buildupRate, 0.0,dMin,bi,fcMos)
	#initialize empty arps object
	arp = fitarps.fitarps()
	new = converter.convertfromfront(thb,arp,"absolute")
	print("new arps: ",arp)
	print("arps: ",arp.getEUR(fcMos))
	arpeur = arp.getEUR(fcMos)

	body = {
		"isBase64Encoded": False,
		"headers": {
		 	"Access-Control-Allow-Origin": "*"
		},
		"body": {
			"data": arp
		},
		"statusCode": 200,
		"msg" : '', # not QServer
	}

	return {
		"isBase64Encoded": False,
		"statusCode": 200,
		"headers": { "token": "get-basins", "Access-Control-Allow-Origin":"*"},
		"body": body,
	}

if __name__ == '__main__':
	fcn = arpsfcn()
	rate = fcn.generateRates(986.04, 2.0, 0.57, 0.05,600)
	#print("rates",fcn.rates)
	print("cums", fcn.cums)

	#print("rates",fcn.rates)
	#print("cums", fcn.cums)
	print(fcn.getDouble("eur"))
