from __future__ import print_function
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession,SQLContext
from pyspark.sql import Row
from pyspark.sql.types import *
from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD,RidgeRegressionWithSGD,LassoWithSGD
from pyspark.streaming import StreamingContext
from pyspark.mllib.linalg import Matrices
from pyspark.mllib.linalg.distributed import BlockMatrix
import matplotlib.path as mplPath
import numpy as np
import time,math,re,random
from scipy import ndimage

spark = SparkSession.builder.master("spark://ukko050:7077").\
            appName("exercise_set_4").getOrCreate()
sc = spark.sparkContext


reference = [["Black","White","Red","Green","Blue","Yellow","Cyan","Magenta",
			"Silver","Gray","Maroon","Olive","Green","Purple","Teal","Navy"],
			np.mat([[0,0,0],[255,255,255],[255,0,0],[0,255,0],[0,0,255],[255,255,0],[0,255,255],[255,0,255],
					[192,192,192],[128,128,128],[128,0,0],[128,128,0],[0,128,0],[128,0,128],[0,128,128],
					[0,0,128]])]
imglist = sc.textFile("./jpg/files.txt",64) #reading file name
bc = sc.broadcast(reference)
dominant = imglist.map(lambda name: (name,"/cs/work/home/jxhou/Spark/jpg/"+name))\
				.map(lambda (name,path):(name, ndimage.imread(path)))\
				.map(lambda (name,img) :(name,img.reshape(img.shape[0]*img.shape[1],img.shape[2]).tolist()))\
				.map(lambda (name,rgb):[(name,rgb[i]) for i in range(len(rgb))]).flatMap(lambda x:x)\
				.map(lambda (name,rgb):(name,
		bc.value[0][np.sqrt(np.squeeze(np.asarray(np.square(bc.value[1]-np.mat(rgb))).sum(1))).argmin()]))\
				.map(lambda key:(key,1)).reduceByKey(lambda a,b:a+b)\
				.map(lambda (key,count):(key[0],(key[1],count)))\
				.reduceByKey(lambda a,b: ((a[1]>=b[1] and a) or b))\
				.map(lambda (key,values):(key,values[0]))
'''
reading images
transfer into pixel array
reshape and transfer into rgb array
marking image name and spliting rgb array
find nearest neighbour(Euclidean distabce) and replace rgb with color
count fraquancy of each color for each image
filter our dominant color of each image and re-arrage data presentation
'''			
reference = [["maroon","dark red","brown","firebrick","crimson","red","tomato","coral",
"indian red","light coral","dark salmon","salmon","light salmon","orange red","dark orange",
"orange","gold","dark golden rod","golden rod","pale golden rod","dark khaki","khaki","olive",
"yellow","yellow green","dark olive green","olive drab","lawn green","chart reuse","green yellow",
"dark green","green","forest green","lime","lime green","light green","pale green","dark sea green",
"medium spring green","spring green","sea green","medium aqua marine","medium sea green","light sea green",
"dark slate gray","teal","dark cyan","aqua","cyan","light cyan","dark turquoise","turquoise","medium turquoise",
"pale turquoise","aqua marine","powder blue","cadet blue","steel blue","corn flower blue","deep sky blue",
"dodger blue","light blue","sky blue","light sky blue","midnight blue","navy","dark blue","medium blue",
"blue","royal blue","blue violet","indigo","dark slate blue","slate blue","medium slate blue","medium purple",
"dark magenta","dark violet","dark orchid","medium orchid","purple","thistle","plum","violet","magenta / fuchsia",
"orchid","medium violet red","pale violet red","deep pink","hot pink","light pink","pink","antique white",
"beige","bisque","blanched almond","wheat","corn silk","lemon chiffon","light golden rod yellow","light yellow",
"saddle brown","sienna","chocolate","peru","sandy brown","burly wood","tan","rosy brown","moccasin","navajo white",
"peach puff","misty rose","lavender blush","linen","old lace","papaya whip","sea shell","mint cream","slate gray",
"light slate gray","light steel blue","lavender","floral white","alice blue","ghost white","honeydew","ivory",
"azure","snow","black","dim gray / dim grey","gray / grey","dark gray / dark grey","silver","light gray / light grey",
"gainsboro","white smoke","white"],
np.mat([[128,0,0],[139,0,0],[165,42,42],[178,34,34],[220,20,60],[255,0,0],[255,99,71],[255,127,80],[205,92,92],
[240,128,128],[233,150,122],[250,128,114],[255,160,122],[255,69,0],[255,140,0],[255,165,0],[255,215,0],[184,134,11],
[218,165,32],[238,232,170],[189,183,107],[240,230,140],[128,128,0],[255,255,0],[154,205,50],[85,107,47],[107,142,35],
[124,252,0],[127,255,0],[173,255,47],[0,100,0],[0,128,0],[34,139,34],[0,255,0],[50,205,50],[144,238,144],[152,251,152],
[143,188,143],[0,250,154],[0,255,127],[46,139,87],[102,205,170],[60,179,113],[32,178,170],[47,79,79],[0,128,128],
[0,139,139],[0,255,255],[0,255,255],[224,255,255],[0,206,209],[64,224,208],[72,209,204],[175,238,238],[127,255,212],
[176,224,230],[95,158,160],[70,130,180],[100,149,237],[0,191,255],[30,144,255],[173,216,230],[135,206,235],[135,206,250],
[25,25,112],[0,0,128],[0,0,139],[0,0,205],[0,0,255],[65,105,225],[138,43,226],[75,0,130],[72,61,139],[106,90,205],
[123,104,238],[147,112,219],[139,0,139],[148,0,211],[153,50,204],[186,85,211],[128,0,128],[216,191,216],[221,160,221],
[238,130,238],[255,0,255],[218,112,214],[199,21,133],[219,112,147],[255,20,147],[255,105,180],[255,182,193],
[255,192,203],[250,235,215],[245,245,220],[255,228,196],[255,235,205],[245,222,179],[255,248,220],[255,250,205],
[250,250,210],[255,255,224],[139,69,19],[160,82,45],[210,105,30],[205,133,63],[244,164,96],[222,184,135],[210,180,140],
[188,143,143],[255,228,181],[255,222,173],[255,218,185],[255,228,225],[255,240,245],[250,240,230],[253,245,230],
[255,239,213],[255,245,238],[245,255,250],[112,128,144],[119,136,153],[176,196,222],[230,230,250],[255,250,240],
[240,248,255],[248,248,255],[240,255,240],[255,255,240],[240,255,255],[255,250,250],[0,0,0],[105,105,105],
[128,128,128],[169,169,169],[192,192,192],[211,211,211],[220,220,220],[245,245,245],[255,255,255]])]
