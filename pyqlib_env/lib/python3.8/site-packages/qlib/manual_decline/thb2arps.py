# keep the header import when update content
# it is depends on qlib structure as prioprity
import math as math
import qlib.manual_decline
from qlib.manual_decline.fitthb import fitthb as fitthb
from qlib.manual_decline import fitarps as fitarps
from qlib.manual_decline import thbfcn as thbfcn

class thb2arps:
	test = "myval"
	def __init__(self):
		print("initializing")

		self.test = "myval2"

	def __str__(self):
		return str(self.__dict__)

	def calculateRateValueArps(self,q,b,d,t):
		dnom = float((math.pow(1.0 - d, -b) -1.0) / b / 12.0)
		qn = float(q / math.pow((1.0 + b * dnom * t),(1.0 / b)))
		return qn

	def convertfromfront(self,thb, arps, method = None):
		arps.dMin = thb.dMin
		ip1 = 0.0
		di1 = 0.0
		ip2 = 0.0
		tmo = 0.0
		di2 = 0.0
		t1 = 0.0
		t2 = 0.0
		b1 = thb.bi
		b2 = thb.bf

		if (thb.fcMos == 0) :
			thb.fcMos = 600
		if (thb.t <= 0):
			tSwitch = thb.t

		#Get a time stream
		tSeries = []
		tSeries.insert(0,0.0)
		tSeries.insert(1,15.0)
		#tSeries[0] = 0.0
		#tSeries[1] = 15.0
		#for( int i = 2; i<thb.fcMos; i++ )
		for i in range(2,thb.fcMos):
			tSeries.insert(i,15.0 + (30.4 * (i - 1)))

		#Calculating THB stream
		thbf = thbfcn.thbfcn()
		thbf.generateRatesTHB(thb.q,thb.bi,thb.bf,thb.di,tSwitch,0.0,tSeries)
		rates = thbf.getArray("rates")

		arps.buildupRate = thb.buildupRate
		arps.buildupMonths = thb.buildupMonths
		if (arps.buildupMonths == 0.0):
			arps.buildupRate = 0.0

		if (method.lower() == "absolute"):
			#first leg
			ip1 = thb.q
			tmo = thb.t * 2.5 / 30.4
			dnoml = float((1 - math.pow((rates[0] / rates[1]), thb.bi)) / (-thb.bi * (1.0 *0.5) / 12.0))
			di1 = min((1 - math.pow((dnoml * thb.bi + 1), (1 / -thb.bi))),0.99999)
			t1 = tmo

			#second leg
			#Match point
			qSwitch = self.calculateRateValueArps(thb.q, thb.bi, di1, (tmo))
			ip2 = qSwitch
			tDelay = float(60)
			iMatch = int(round(min((tmo) + tDelay, thb.fcMos)))
			qMatch = rates[iMatch]
			tMatch = iMatch - 0.5
			#Calculate decline from match point
			dnom2 = (math.pow((qSwitch / qMatch), thb.bf)-1) / (thb.bf * (tMatch - (tmo) + 0.0)) * 12.0
			di2 = min(round((1 - math.pow((dnom2 / 12.0 * thb.bf * 12 + 1),(1 / -thb.bf)))* 100000.0)/ 100000.0, 0.99999)
			t2 = thb.fcMos - (tmo) - thb.buildupMonths
		else:
			#first leg
			ip1 = thb.q
			dnoml = float((1 - math.pow((rates[0] / rates[1]), thb.bi)) / (-thb.bi * (1.0 * 0.5) / 12.0))
			di1 = min((1 - math.pow((dnoml * thb.bi + 1),(1 / -thb.bi))),0.99999)
			tmo = thb.t * 2.5 / 30.4
			t1 = math.floor(tmo)
			tx = float(tmo % 1.0)

			#second leg
			#match point
			qSwitch = float(self.calculateRateValueArps(thb.q,thb.bi, di1,t1))
			ip2 = qSwitch
			tdelay = float(60)
			iMatch = int(round(min(t1 + tDelay, thb.fcMos)) - 1)
			qMatch = float(rates[iMatch])
			tMatch = float(iMatch - 0.5)
			dnom2 = (math.pow((qSwitch / qMatch), thb.bf) - 1) / (thb.bf * (tMatch - t1 + 0.0)) * 12.0
			di2 = min(Math.round((1 - math.pow((dnom2 / 12.0 * thb.bf * 12 + 1), (1 / -thb.bf))) * 100000.0) / 100000.0, 0.99999)
			t2 = thb.fcMos - t1 - thb.buildupMonths

			if (t1 == 0.0):
				ip1 = ip2
				b1 = thb.bf
				di1 = di2
				t1 = t2
				ip2 = 0.0
				b2 = 0.0
				di2 = 0.0
				t2 = 0.0
		if (math.isnan(ip1)):
			ip1 = 0.0
		if(math.isnan(ip2)):
			ip2 = 0.0
		if(math.isnan(di1)):
			di1 = 0.0
		if(math.isnan(di2)):
			di2 = 0.0
		t1 = max(t1,0)
		t2 = max(t2,0)

		arps.q1 = ip1
		arps.b1 = b1
		arps.d1 = di1
		arps.t1 = t1
		arps.q2 = ip2
		arps.b2 = b2
		arps.d2 = di2
		arps.t2 = t2
		arps._offset = thb._offset
		arps._cumul = thb.getCumul()

def lambda_handler(event, context):
	#if local test
	if (type(event['body']) == dict):
		data = event['body']['data']
		#Else if coming from API gateway
	else:
		data = json.loads(event['body'])
		data = data['data']
	dMin = data["dMin"]
	t = data["t"]
	bi = data["bi"]
	buildupMonths = data["buildupMonths"]
	buildupRate = data["buildupMonths"]
	qi = data["ip"]
	eur = data["eur"]
	fcMos = data["fcMos"]
	di = data["di"]
	bf = data["bf"]
	apikey = event['body']["apiKey"]
	owner = data["ProjectOwner"]

	res = fitthb(qi, bf,di,t,buildupMonths,buildupRate,eur,dMin,bi,fcMos)

	#conn_data = get_engine_string(apikey)
	#res = fitthb(seg_name, segdata, user, owner, projname, descrip, inputs, conn_data)
	print("res: ", res)

	body = {
		"isBase64Encoded": False,
		"headers": {
			"Access-Control-Allow-Origin": "*"
		},
		"body": {
			"data": res
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
	converter = thb2arps()
	#new = converter.calculateRateValueArps(1807.77,1.14,0.49,0)
	#initialize thb object
	thb = fitthb.fitthb(q =986.04, bf = 1.26,di = 0.57,t = 0.0,buildupMonths = 25,buildupRate = 679.54,eur =0,dMin = 0.05,bi = 2.0,fcMos = 600)
	#print(thb)
	#initialize empty arps object
	arp = fitarps.fitarps()
	new = converter.convertfromfront(thb,arp,"absolute")
	print("new: ", new)
	print("arps: ",arp)
	print("EUR : ",arp.getEUR(600))
