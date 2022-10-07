from strategy.common_str import *

class Prob_str(Strategy):
	def __init__(self):
		super().__init__()
	
	def jouer(self):
		nb=0
		while not self.bat.victoire():
			pos=self.prop_position()
			p=self.bat.joue(pos)
			nb+=1
			if p==1:
				nb+=self.snipe(pos)
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
			return self.snipe_dir(pos,direction)+1
		return 1
	
	def affiche_stat(self,fois):
		data=[]
		for i in range(fois):
			self.bat.reset()
			data.append(self.jouer())
		plt.hist(data,bins=100)
		plt.xlabel("nb steps")
		plt.ylabel("frequency")
		plt.savefig('./figures/prob.jpg')
		
	def prop_position(self):
		prop_mat=np.zeros((10,10))
		for ships in self.bat.ships_remain():
			s=[]
			for attempt in range(100):
				pos_mat=np.zeros((10,10))
				pos=super().generer_alea()
				dir=np.random.randint(2)+1
				if Grille.peut_placer(self.bat.record,ships,pos,dir):
					length = Grille.bat_longueur(ships)
					if dir == 1:
						for i in range(pos[0], pos[0]+length):
							pos_mat[i][pos[1]] = 1
					if dir == 2:
						for i in range(pos[1], pos[1]+length):
							pos_mat[pos[0]][i] = bateau
				if pos_mat not in s:
					s.append(pos_mat)
			for mat_temp in s:
				prop_mat+=s
		return np.unravel_index(np.argmax(prop_mat),prop_mat.shape)
		