"""
Student Name : Hou, Jue	
Student ID   : 014695647
"""
class Book:
	isbn = ""
	title = ""
	author = ""
	year = 0
	publisher = ""
	def __init__(self,data):
		values = data.split(";")
		self.isbn = values[0].replace('"','')
		self.title = values[1].replace('"','')
		self.author = values[2].replace('"','')
		self.year = int(values[3].replace('"',''))
		self.publisher = values[4].replace('"','')

class BookRating:
	user_id = ""
	isbn = ""
	rating = -1
	def __init__(self,data):
		values = data.split(";")
		self.user_id = values[0].replace('"','')
		self.isbn = values[1].replace('"','')
		self.rating = int(values[2].replace('"',''))

class User:
	user_id = ""
	location = ""
	age = -1
	def __init__(self,data):
		values = data.split(";")
		self.user_id = values[0].replace('"','')
		self.location = values[1].replace('"','')
		str_age = values[2].replace('"','')
		if(str_age!="NULL"):
			self.age = int(str_age)
		
class RatingUser:
	user_id = ""
	location = ""
	age = -1
	rating = 0
	def __init__(self,user_id,location,age,rating):
		self.user_id = user_id
		self.location = location
		self.age = age
		self.rating = rating