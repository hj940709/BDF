"""
Student Name : Hou, Jue	
Student ID   : 014695647
"""
from __future__ import print_function
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql.types import *
from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD,RidgeRegressionWithSGD,LassoWithSGD
from pyspark.streaming import StreamingContext
from pyspark.mllib.linalg import Matrix
import matplotlib.path as mplPath
import numpy as np
import time,math,re,random,_thread
class ExerciseSet4(object):
	"""
	Big Data Frameworks Exercises
	https://www.cs.helsinki.fi/courses/582740/2017/k/k/1
	"""

	def __init__(self):
		"""
		Initializing Spark Conf and Spark Context here
		Some Global variables can also be initialized
		"""
		self.spark = SparkSession.builder.master("spark://ukko050:7077").\
						appName("exercise_set_4").getOrCreate()
		self.spark_context = self.spark.sparkContext
		# Have global variables here if you wish
		# self.global_variable = None

	def exercise_1(self):
		"""
		# Write your Docstring here
		"""
		sc = self.spark_context
		df = self.spark.read.json("./countries_and_coordinates.json")
		data = df.rdd.map(lambda row: (row.name,mplPath.Path(row.coordinates)))
		coordinates = [(24.9345427,60.2576009),
						(-73.983803,40.7690327),
						(116.3949606,39.9163488),
						(-0.1407132,51.4955324),
						(24.963352,60.2039806),
						(131.0283696,-25.3456376)]
		bc = sc.broadcast(coordinates)
		reduced_data = data.filter(lambda country: country[1].contains_points(bc.value).any()).cache()
		result = [(reduced_data.filter(lambda country: country[1].contains_point(bc.value[i]))\
						.map(lambda country:country[0]).collect(),bc.value[i]) for i in range(len(bc.value))]
		print result
		'''
		[(['Finland'], (24.9345427, 60.2576009)),
		 (['United States of America'], (-73.983803, 40.7690327)),
		 (['China'], (116.3949606, 39.9163488)),
		 (['United Kingdom'], (-0.1407132, 51.4955324)),
		 (['Finland'], (24.963352, 60.2039806)),
		 (['Australia'], (131.0283696, -25.3456376))]
		'''
		return None

	def exercise_2(self):
		"""
		# Write your Docstring here
		"""
		sc = self.spark_context
		file = sc.textFile("./carat-context-factors-percom.csv")
		energyRate,batteryHealth,batteryTemperature,batteryVoltage,\
			cpuUsage,distanceTraveled,mobileDataActivity,mobileDataStatus,\
			mobileNetworkType,networkType,roamingEnabled,screenBrightness,\
			wifiLinkSpeed,wifiSignalStrength = [i for i in range(0,14)]
		data = file.map(lambda line: line.split(";")).map(lambda line:
			(float(line[energyRate]),line[batteryHealth],
			float(line[batteryTemperature]),float(line[batteryVoltage]),
			float(line[cpuUsage]),float(line[distanceTraveled]),
			line[mobileDataActivity],line[mobileDataStatus],
			line[mobileNetworkType],line[networkType],
			float(line[roamingEnabled]),float(line[screenBrightness]),
			float(line[wifiLinkSpeed]),float(line[wifiSignalStrength])))
		data = data.filter(lambda x:((x[screenBrightness]==-1 or(x[screenBrightness]>=0 and x[screenBrightness]<=255)) and\
							(x[cpuUsage]>=0 and x[cpuUsage]<=1) and\
							(x[distanceTraveled]>=0) and\
							(x[wifiSignalStrength]>-100 and x[wifiSignalStrength]<0) and\
							(x[batteryTemperature]>=0)))
		data = data.map(lambda x:LabeledPoint(x[energyRate],
					[x[cpuUsage],x[screenBrightness], x[wifiSignalStrength], x[batteryTemperature]]))
		train,test = data.randomSplit([4,1])

		lr = LinearRegressionWithSGD.train(train,iterations=100,step=1e-4,intercept=False)
		print lr#(weights=[4.05918718288e-07,2.01710179227e-05,-3.39410603521e-05,1.70383825251e-05], intercept=0.0)
		rr = RidgeRegressionWithSGD.train(train,iterations=100,step=1e-4,intercept=False)
		print rr#(weights=[4.05918453228e-07,2.0170994023e-05,-3.39410381473e-05,1.70383716836e-05], intercept=0.0)
		l = LassoWithSGD.train(train,iterations=100,step=1e-4,intercept=False)
		print l#(weights=[0.0,1.96629057526e-05,-3.29054093642e-05,1.56445907401e-05], intercept=0.0)
		valuesAndPreds = test.map(lambda p: (p.label,lr.predict(p.features),
								rr.predict(p.features),l.predict(p.features)))
		count = valuesAndPreds.count()
		MSE = valuesAndPreds.map(lambda (v,lrp,rrp,lp): ((v - lrp)**2/count,
									(v - rrp)**2/count,(v - lp)**2/count))\
							.reduce(lambda a,b:(a[0]+b[0],a[1]+b[1],a[2]+b[2]))
		print MSE #(4.7634385303075644e-05, 4.7634387065855108e-05, 4.7873793406702168e-05)
		return None

	def exercise_3(self):
		"""
		# Write your Docstring here
		"""
		sc = self.spark_context
		def timeout(ssc):
			time.sleep(120)
			ssc.stop(stopSparkContext=False, stopGraceFully=True)
			
		#1
		ssc = StreamingContext(sc, 1)
		stream = ssc.socketTextStream("ukko054", 8890)
		result=stream.flatMap(lambda line: line.split(",")).map(lambda x:float(x))\
					.reduce(lambda a,b:a+b).map(lambda x:x/100)
		result.pprint()
		thread.start_new_thread(timeout,(ssc,))
		ssc.start()
		#ssc.awaitTermination(120)
		
		#2
		def updateFunc(new_values, last_sum):
			return sum(new_values) + (last_sum or 0)
		ssc = StreamingContext(sc, 1)
		stream1 = ssc.socketTextStream("ukko054", 8890)
		stream2 = ssc.socketTextStream("ukko054", 8890)
		stream3 = ssc.socketTextStream("ukko054", 8890)
		stream4 = ssc.socketTextStream("ukko054", 8890)
		stream5 = ssc.socketTextStream("ukko054", 8890)
		result1=stream1.flatMap(lambda line: line.split(",")).map(lambda x:(float(x),1))
		result2=stream2.flatMap(lambda line: line.split(",")).map(lambda x:(float(x),1))
		result3=stream3.flatMap(lambda line: line.split(",")).map(lambda x:(float(x),1))
		result4=stream4.flatMap(lambda line: line.split(",")).map(lambda x:(float(x),1))
		result5=stream5.flatMap(lambda line: line.split(",")).map(lambda x:(float(x),1))					
		all = result1.union(result2).union(result3).union(result4).union(result5).updateStateByKey(updateFunc)
		filtered = all.filter(lambda x:x[1]>1)
		filtered .pprint()
		ssc.checkpoint("checkpoint")
		thread.start_new_thread(timout,(ssc,))
		ssc.start()
		#ssc.awaitTermination(120)
		
		#3
		ssc = StreamingContext(sc, 1)
		stream1 = ssc.socketTextStream("ukko054", 8890)
		stream2 = ssc.socketTextStream("ukko054", 8890)
		stream3 = ssc.socketTextStream("ukko054", 8890)
		stream4 = ssc.socketTextStream("ukko054", 8890)
		stream5 = ssc.socketTextStream("ukko054", 8890)
		result1=stream1.flatMap(lambda line: line.split(",")).map(lambda x: float(x))
		result2=stream2.flatMap(lambda line: line.split(",")).map(lambda x: float(x))
		result3=stream3.flatMap(lambda line: line.split(",")).map(lambda x: float(x))
		result4=stream4.flatMap(lambda line: line.split(",")).map(lambda x: float(x))
		result5=stream5.flatMap(lambda line: line.split(",")).map(lambda x: float(x))
		
		def is_prime(n):
			if n == 1:
				return False
			for i in range(2, int(math.sqrt(n))+1):
				if n % i == 0:
					return False
			return True
		
		all = result1.union(result2).union(result3).union(result4).union(result5).filter(lambda x:is_prime(x))
		all.pprint()
		thread.start_new_thread(timout,(ssc,))
		ssc.start()
		#ssc.awaitTermination(120)
		return None

	def exercise_4(self):
		"""
		# Write your Docstring here
		"""
		sc = self.spark_context
		row = sc.parallelize([i for i in range(1000)])
		A = row.map(lambda x: [(x,j) for j in range(10000)]).flatMap(lambda entry:entry)\
							.map(lambda entry:(entry,random.random()))
		A_T = A.map(lambda (entry,value):((entry[1],entry[0]),value))

		id = sc.parallelize([i for i in range(10)])
		A1 = id.map(lambda x: (0,x)).map(lambda entry:(entry,
				np.mat([[random.random() for i in range(1000)] for j in range(1000)])))
		A2 = id.map(lambda x: (x,0)).map(lambda entry:(entry,
				np.mat([[random.random() for i in range(1000)] for j in range(1000)])))
		A2_T = A2.map(lambda (entry,value):((entry[1],entry[0]),value))
		A = A1.union(A2_T).reduceByKey(lambda a,b: a.dot(b)).reduce(lambda a,b: a[1]+b[1])
		return None

	def exercise_5(self):
		"""
		# Write your Docstring here
		"""
		# self.spark_context
		return None

	def exercise_6(self):
		"""
		# Write your Docstring here
		"""
		# self.spark_context
		return None
	
	def stop(self):
		self.spark_context.stop()

if __name__ == "__main__":
	EXERCISESET = ExerciseSet4()
	EXERCISESET.exercise_1()
	EXERCISESET.exercise_2()
	EXERCISESET.exercise_3()
	EXERCISESET.exercise_4()
	EXERCISESET.exercise_5()
	EXERCISESET.exercise_6()
	EXERCISESET.stop()
	