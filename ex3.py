"""
Student Name : Hou, Jue	
Student ID   : 014695647
"""
from __future__ import print_function
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql.types import *
import time,math,re
class ExerciseSet3(object):
	"""
	Big Data Frameworks Exercises
	https://www.cs.helsinki.fi/courses/582740/2017/k/k/1
	"""

	def __init__(self):
		"""
		Initializing Spark Conf and Spark Context here
		Some Global variables can also be initialized
		"""
		self.spark = SparkSession.builder.master("local").\
						appName("exercise_set_2").\
						config("spark.executor.memory", "2g").getOrCreate()
		self.spark_context = self.spark.sparkContext
		# Have global variables here if you wish
		# self.global_variable = None

	def exercise_1(self):
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
        	int(line[roamingEnabled]),float(line[screenBrightness]),
        	int(line[wifiLinkSpeed]),int(line[wifiSignalStrength])))
		data = data.filter(lambda x:((x[screenBrightness]==-1 or(x[screenBrightness]>=0 and x[screenBrightness]<=255)) and\
							(x[cpuUsage]>=0 and x[cpuUsage]<=1) and\
							(x[distanceTraveled]>=0) and\
							(x[wifiSignalStrength]>-100 and x[wifiSignalStrength]<0) and\
							(x[batteryTemperature]>=0)))
    	def getPearsonCorrelation(data,index_1,index_2):
    	    start = time.time()
			xrdd = data.map(lambda x:x[index_1])
			yrdd = data.map(lambda x:x[index_2])
			mx = xrdd.mean()
			my = yrdd.mean()
			x_mx = xrdd.map(lambda x:x-mx)
			y_my = yrdd.map(lambda y:y-my)
			both = x_mx.zip(y_my)
			upper = both.map(lambda array: array[0]*array[1]).sum()
			lowerx,lowery = both.map(lambda array:(math.pow(array[0],2),math.pow(array[1],2))).\
								reduce(lambda a,b:(a[0]+b[0],a[1]+b[1]))
			correlation = upper/(math.sqrt(lowerx) * math.sqrt(lowery))
			end = time.time()
			print "Correclation:",correlation,"Time:",end-start
			
		getPearsonCorrelation(data,energyRate,cpuUsage)
		getPearsonCorrelation(data,energyRate,screenBrightness)
		getPearsonCorrelation(data,energyRate,wifiLinkSpeed)
		getPearsonCorrelation(data,energyRate,wifiSignalStrength)
		
		return None

	def exercise_2(self):
		"""
		# Write your Docstring here
		"""
		#a
		caratDF = spark.read.csv("./carat-context-factors-percom.csv",sep=";")
		caratDF = caratDF.toDF("energyRate","batteryHealth","batteryTemperature",
		"batteryVoltage","cpuUsage","distanceTraveled",
		"mobileDataActivity","mobileDataStatus","mobileNetworkType",
		"networkType","roamingEnabled","screenBrightness",
		"wifiLinkSpeed","wifiSignalStrength")
		#b
		caratDF = caratDF.withColumn("energyRate",caratDF['energyRate'].cast(DoubleType())).\
			withColumn("batteryTemperature",caratDF['batteryTemperature'].cast(DoubleType())).\
			withColumn("batteryVoltage",caratDF['batteryVoltage'].cast(DoubleType())).\
			withColumn("cpuUsage",caratDF['cpuUsage'].cast(DoubleType())).\
			withColumn("distanceTraveled",caratDF['distanceTraveled'].cast(DoubleType())).\
			withColumn("roamingEnabled",caratDF['roamingEnabled'].cast(DoubleType())).\
			withColumn("screenBrightness",caratDF['screenBrightness'].cast(DoubleType())).\
			withColumn("wifiLinkSpeed",caratDF['wifiLinkSpeed'].cast(DoubleType())).\
			withColumn("wifiSignalStrength",caratDF['wifiSignalStrength'].cast(DoubleType()))
		#c
		caratDF.select("batteryTemperature").distinct().count()
		caratDF.select("batteryVoltage").distinct().count()
		caratDF.filter("batteryVoltage<2 or batteryVoltage>4.35").count()
		caratDF.filter("batteryTemperature<0 or batteryTemperature>50").count()
		#d
		carat = caratDF.filter("cpuUsage>=0 and cpuUsage<=1 and (screenBrightness=-1 or screenBrightness>=0 and screenBrightness<=255)"+
							" and wifiSignalStrength>-100 and wifiSignalStrength<0 and distanceTraveled>=0 and batteryTemperature>=0").\
							select("energyRate","cpuUsage","screenBrightness","wifiSignalStrength","wifiLinkSpeed")
		#e
		start = time.time()
		corr = carat.corr("energyRate","cpuUsage")
		stop = time.time()
		print corr,stop-start
		start = time.time()
		corr = carat.corr("energyRate","screenBrightness")
		stop = time.time()
		print corr,stop-start
		start = time.time()
		corr = carat.corr("energyRate","wifiLinkSpeed")
		stop = time.time()
		print corr,stop-start
		start = time.time()
		corr = carat.corr("energyRate","wifiSignalStrength")
		stop = time.time()
		print corr,stop-start
		return None

	def exercise_3(self):
		"""
		# Write your Docstring here
		"""
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
			int(line[roamingEnabled]),float(line[screenBrightness]),
			int(line[wifiLinkSpeed]),int(line[wifiSignalStrength])))
		filtered_data=data.filter(lambda x:x[screenBrightness]>-1 and x[screenBrightness]<=255).map(lambda x:(x[screenBrightness],x[energyRate]))
		number = filtered_data.count()
		time1 = time.time()
		grouped_data = filtered_data.groupByKey()
		array1 = grouped_data.mapValues(sum).collect()
		time2 = time.time()
		reduced_data = filtered_data.reduceByKey(lambda a,b:a+b)
		array2 = reduced_data.collect()
		'''
		time3 = time.time()
		reduced_data = filtered_data.partitionBy(50,partitionFunc = lambda x: x%50).reduceByKey(lambda a,b:a+b)
		array3 = reduced_data.collect()
		time4 = time.time()
		reduced_data = filtered_data.partitionBy(50,partitionFunc = lambda x: x/(number/50)).reduceByKey(lambda a,b:a+b)
		array4 = reduced_data.collect()
		time5 = time.time()
		'''
		print len(array1),len(array2)#,len(array3),len(array4)
		return None

	def exercise_4(self):
		"""
		# Write your Docstring here
		"""
		stopword = sc.textFile("./stopwords.txt").collect()
		raw = sc.wholeTextFiles("./onlytxt")
		files = raw.mapValues(lambda file: file.split(" "))
		bc = sc.broadcast(stopword)
		def filter(bag,stopwords):
			result = []
			for word in bag:
				filtered_word = re.sub(r'(\d+|\W+)','',word)
				if(len(filtered_word)>1 and (not filtered_word in stopwords)):
					result.append(filtered_word)
			return [(k,result.count(k)) for k in set(result)]	
		files = files.map(lambda file:(file[0],filter(file[1],bc.value)))
		df = files.flatMap(lambda file: file[1]).map(lambda word:(word[0],1)).reduceByKey(lambda a,b:a+b)
		D=raw.count()
		idf = df.map(lambda x: (x[0],math.log((D+1)/(x[1]+1))))
		tf = files.map(lambda file: [(word[0],(file[0],word[1])) for word in file[1]]).flatMap(lambda x:x)
		tfidf = tf.join(idf).map(lambda x:(x[0],x[1][0][0],x[1][0][1]*x[1][1]))
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
	EXERCISESET = ExerciseSet3()
	EXERCISESET.exercise_1()
	EXERCISESET.exercise_2()
	EXERCISESET.exercise_3()
	EXERCISESET.exercise_4()
	EXERCISESET.exercise_5()
	EXERCISESET.exercise_6()
	EXERCISESET.stop()
	