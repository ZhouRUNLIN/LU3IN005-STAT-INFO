import numpy
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class Grille:
    def __init__(self):
        """
        *0-9
        Supposons que (0,0) est le point le plus haut et le plus gauche.
        """
        self.grille=numpy.zeros((10,10))

    def bat_longueur(self,bateau):
        if bateau == 1:
            length=5
        elif bateau == 2:
            length=4
        elif bateau == 3 or bateau == 4:
            length=3
        elif bateau == 1:
            length=2
        else:
            length=0
        return length

    def peut_placer(self, grille, bateau, position, direction):
        """
        bateau int : type des bateaux
        position (int,int) : position de tête(gauche ou haut)
        direction int : 1 pour horizontale et 2 pour verticale

        return True si possible, False sinon
        """
        # tester si le point de tête est dans la grille
        if position[0]<0 or position[0]>9 or position[1]<0 or position[1]>9:
            return False

        # calculer le longueur de bateau
        length = self.bat_longueur(bateau)
        if length==0:
            return False

        if direction==1:
           if position[0]+length<9:
                for i in range(position[0], position[0]+length):
                    if self.grille[i][position[1]] != 0:
                       return False
                return True
        
        if direction==2:
           if position[1]+length<9:
                for i in range(position[1], position[1]+length):
                    if self.grille[position[0]][i] != 0:
                       return False
                return True
        return False
    
    def place(self, grille, bateau, position, direction):
        """
        bateau int : type des bateaux
        position (int,int) : position de tête(gauche ou haut)
        direction int : 1 pour horizontale et 2 pour verticale

        return True si cette opération est validé, False sinon
        """
        if not self.peut_placer(grille, bateau, position, direction):
            return False
        
        length = self.bat_longueur(bateau)

        if direction == 1:
            for i in range(position[0], position[0]+length):
                self.grille[i][position[1]] = bateau
        
        if direction == 2:
            for i in range(position[1], position[1]+length):
                self.grille[position[0]][i] = bateau
        return True

    def generer_position(self):
        """
        return ((int,int),int) : (position,direction)
        """
        x=numpy.random.randint(0,10)
        y=numpy.random.randint(0,10)
        d=numpy.random.randint(1,3)
        return ((x,y),d)

    def place_alea(self, grille, bateau):
        """
        grille int**2 : le champ de grille
        bateau int : type des bateaux

        return None
        """
        pos,dir=self.generer_position()
        while not self.peut_placer(grille,bateau,pos,dir):
            pos,dir=self.generer_position()
    
    def affiche(self, grille):
        """
        grille int**2 : le champ de grille

        return None
        """
        fig = plt.figure()
        ax2 = fig.add_subplot(122)
        ax2.imshow(grille, interpolation='nearest', cmap=cm.Greys_r)
        plt.show()
    
    def eq(self, grilleA, grilleB):
        """
        grilleA int**2 : le champ de grille
        grilleB int**2 : le champ de grille

        return True si grilleA==grilleB, false sinon
        """
        return numpy.array_equal(grilleA,grilleB)

    def genere_grille(self):
        """
        """
        for i in range(1,6):
            self.place_alea(self.grille,i)

g0=Grille()
print(g0.place_alea(g0.grille,1))
g0.affiche(g0.grille)


