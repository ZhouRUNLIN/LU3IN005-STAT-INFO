from strategy.common_str import *

class Heur_str(Strategy):
	def __init__(self):
		super().__init__()
	
	def jouer(self):
		nb=0
		while not self.bat.victoire():
			pos=super().generer_alea()
			p=self.bat.joue(pos)
			nb+=1
			if p==1:
				nb+=self.snipe(pos)
		self.bat.reset()
		return nb
	
	def snipe(self,position):
		nb=0
		for dir in [(1,0),(-1,0),(0,1),(0,-1)]:
			nb+=self.snipe_dir(position,dir)
		return nb
		
	def snipe_dir(self,position,direction):
		pos=(position[0]+direction[0],position[1]+direction[1])
		if pos[0]<0 or pos[0]>9 or pos[1]<0 or pos[1]>9:
			return 0
		if self.bat.record[pos[0]][pos[1]]!=0:
			return 0
		if self.bat.victoire():
			return 0
		p=self.bat.joue(pos)
		if p==1:
			self.bat.count+=1
			return self.snipe_dir(pos,direction)+1
		return 1