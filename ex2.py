"""
Student Name : Hou, Jue	
Student ID   : 014695647
"""
from __future__ import print_function
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql.types import *
from ex2class import Book,BookRating,User,RatingUser#need to be corrected to match the filename
import time
class ExerciseSet2(object):
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
		
		start = time.time()
		
		bookfile = sc.textFile("./BX-Books.csv")
		bookratingfile = sc.textFile("./BX-Book-Ratings.csv")
		userfile = sc.textFile("./BX-Users.csv")
		books = bookfile.map(lambda line: Book(line))
		bookratings = bookratingfile.map(lambda line: BookRating(line))
		users = userfile.map(lambda line: User(line))
		books.saveAsPickleFile("./books")
		bookratings.saveAsPickleFile("./bookratings")
		users.saveAsPickleFile("./users")
		
		end = time.time()
		print("Execution Time(Without Specifying Partions):",end-start)
		
		start = time.time()
		
		bookfile = sc.textFile("./BX-Books.csv",5)
		bookratingfile = sc.textFile("./BX-Book-Ratings.csv",5)
		userfile = sc.textFile("./BX-Users.csv",5)
		books = bookfile.map(lambda line: Book(line))
		bookratings = bookratingfile.map(lambda line: BookRating(line))
		users = userfile.map(lambda line: User(line))
		books.saveAsPickleFile("./books5")
		bookratings.saveAsPickleFile("./bookratings5")
		users.saveAsPickleFile("./users5")
		
		end = time.time()
		print("Execution Time(Specifying 5 Partions):",end-start)
		return None

	def exercise_2(self):
		"""
		# Write your Docstring here
		"""
		sc = self.spark_context
		books = sc.pickleFile("./books")
		bookratings = sc.pickleFile("./bookratings")
		users = sc.pickleFile("./users")

		id_users = users.map(lambda user:(user.user_id,user))
		userid_bookratings = bookratings.map(lambda bookrating:(bookrating.user_id,bookrating))
		isbn_userratings = id_users.join(userid_bookratings).\
							map(lambda x: (x[1][1].isbn,set([RatingUser(x[1][0].user_id,
															x[1][0].location,x[1][0].age,x[1][1].rating)]))).\
							reduceByKey(lambda a,b:a.union(b))
		result = books.map(lambda book:(book.isbn,book)).\
					join(isbn_userratings).\
					map(lambda x:(x[0],x[1][0].title,x[1][0].author,x[1][0].year,x[1][0].publisher,x[1][1]))		
		result.saveAsPickleFile("./result")
		return None

	def exercise_3(self):
		"""
		# Write your Docstring here
		"""
		sc = self.spark_context
		ranked_books = sc.pickleFile("./result")
		number = ranked_books.filter(lambda book: book[3]>=1992 and book[3]<=1998).\
					map(lambda book: len(book[5])).sum()	
		print("There are",number,"books in total published between 1992 and 1998")
		return None

	def exercise_4(self):
		"""
		# Write your Docstring here
		"""
		sc = self.spark_context
		books = sc.pickleFile("./books")
		bookratings = sc.pickleFile("./bookratings")
		users = sc.pickleFile("./users")

		id_users = users.map(lambda user:(user.user_id,user))
		userid_bookratings = bookratings.map(lambda bookrating:(bookrating.user_id,bookrating))
		isbn_userratings = id_users.join(userid_bookratings).\
							map(lambda x: (x[1][1].isbn,set([RatingUser(x[1][0].user_id,
															x[1][0].location,x[1][0].age,x[1][1].rating)]))).\
							reduceByKey(lambda a,b:a.union(b))
		result = books.map(lambda book:(book.isbn,book)).\
					join(isbn_userratings).\
					map(lambda x:(x[0],x[1][0].title,x[1][0].author,x[1][0].year,x[1][0].publisher,x[1][1]))
		#result = sc.pickleFile("./result")
		def getAverage(sets):
			sum = 0
			n = 0
			for s in sets:
				if(s.age>0):
					sum = sum + s.age
					n=n+1
			if(n==0): return 0
			return sum/n
		result.map(lambda x: (x[2],getAverage(x[5]))).sortBy(lambda x: x[1],ascending=False).take(20)
		
		return None

	def exercise_5(self):
		"""
		# Write your Docstring here
		"""
		spark = self.spark
		sc = self.spark_context
		bookfile = sc.textFile("./BX-Books.csv")
		bookratingfile = sc.textFile("./BX-Book-Ratings.csv")
		userfile = sc.textFile("./BX-Users.csv")

		booksrow = lambda line: Row(name=line[0],title=line[1],author=line[2],
							year=int(line[3]),publisher=line[4])
		bookratingrow = lambda line: Row(user_id=line[0],isbn=line[1],rating=int(line[2]))
		usersrow = lambda line: Row(user_id=line[0],location=line[1],age=line[2])

		books = bookfile.map(lambda line: line.replace('"','').split(';')).map(booksrow)
		bookratings = bookratingfile.map(lambda line: line.replace('"','').split(';')).map(bookratingrow)
		users = userfile.map(lambda line: line.replace('"','').split(';')).map(usersrow)

		booksdf = spark.createDataFrame(books)
		bookratingsdf = spark.createDataFrame(bookratings)
		usersdf = spark.createDataFrame(users)

		booksdf.write.save("./booksdf")
		bookratingsdf.write.save("./bookratingsdf")
		usersdf.write.save("./usersdf")

		booksdf.filter(booksdf['year'] >= 1992).filter(booksdf['year'] <= 1998).show()
		
		return None

	def stop(self):
		self.spark_context.stop()

if __name__ == "__main__":
	EXERCISESET2 = ExerciseSet2()
	EXERCISESET2.exercise_1()
	EXERCISESET2.exercise_2()
	EXERCISESET2.exercise_3()
	EXERCISESET2.exercise_4()
	EXERCISESET2.exercise_5()
	EXERCISESET2.stop()
	
