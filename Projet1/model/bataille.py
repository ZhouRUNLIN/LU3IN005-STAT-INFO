"""
Partie 3
"""

from model.grille import *

class Bataille:
	def __init__(self):
		self.g1=Grille()
		self.g1.genere_grille()
		self.record = numpy.zeros((10, 10))
		self.count=0
		
	def joue(self, position):
		self.record[position[0]][position[1]]=1
		print(self.record)
		if self.g1.grille[position[0]][position[1]]!=0:
			self.count+=1
			print(position)
			return 1
		return 0
	
	def victoire(self):
		if self.count==17:
			return 1
		return 0
	
	def reset(self):
		self.__init__()
		