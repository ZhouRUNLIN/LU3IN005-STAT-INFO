from strategy.common_str import *

class Alea_str(Strategy):
	def __init__(self):
		super().__init__()
	
	def jouer(self):
		nb=0
		while not self.bat.victoire():
			self.bat.joue(super().generer_alea())
			nb+=1
		self.bat.reset()
		return nb