from ast import Global
import csv
import random
import numpy as np

# Var globales
LARGEUR = 800
HAUTEUR = 600
NB_LIEUX = 1000


class Lieu:
    def __init__(self, nom, x, y):
        self.nom = nom
        self.x = x
        self.y = y
        self.nbr_route = 0

    def distance(self, autre_lieu):
        return ((self.x - autre_lieu.x) ** 2 + (self.y - autre_lieu.y) ** 2) ** 0.5

    def __repr__(self):
        return f"({self.nom},{self.x},{self.y})"


class Graph:
    def __init__(self):
        self.liste_lieux = []
        self.matrice_od = []
        self.route_numerotee = []

    def generer_lieux_aleatoires(self, nb_lieux=NB_LIEUX):
        self.liste_lieux = [
            Lieu(f"L{i}", random.uniform(0, LARGEUR), random.uniform(0, HAUTEUR))
            for i in range(nb_lieux)]
        self.calcul_matrice_cout_od()

    def charger_graph(self, chemin_fichier):
        global NB_LIEUX
        with open(chemin_fichier, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)

            for row in reader:
                x = float(row['x'])
                y = float(row['y'])
                self.liste_lieux.append(Lieu(f"L{row}", x, y))
        NB_LIEUX = len(self.liste_lieux)
        self.calcul_matrice_cout_od()
        

    def calcul_matrice_cout_od(self):
      self.matrice_od = np.zeros((NB_LIEUX, NB_LIEUX))

      for i in range(NB_LIEUX - 1):
          self.matrice_od[i, i] = np.inf
          for j in range(i + 1, NB_LIEUX):
              d = self.liste_lieux[i].distance(self.liste_lieux[j])
              self.matrice_od[i, j] = d
              self.matrice_od[j, i] = d
      self.matrice_od[NB_LIEUX-1, NB_LIEUX-1] = np.inf
      self.distances = self.matrice_od.copy()
      print(self.matrice_od)

    def plus_proche_voisin(self, nom_lieu, lieu_fait):
      self.distances[:,lieu_fait] = np.inf
      j = np.argmin(self.distances[int(nom_lieu[1])])
      return f"L{j}"

    def calcul_distance_route(self, route):
        distance_totale = 0
        for i in range(NB_LIEUX):
            distance_totale += self.matrice_od[int(route[i][1])][int(route[i+1][1])]
        return distance_totale

    def heuristique_voisin(self):
        lieu_fait = [] 
        self.route_numerotee.append(f"L0")
        lieu_fait = 0
        numero = 0
        for tour in range(NB_LIEUX-1):
            voisin = self.plus_proche_voisin(f"L{numero}", lieu_fait)
            lieu_fait = int(voisin[1])
            self.route_numerotee.append(voisin)
            numero = voisin[1]
        self.route_numerotee.append(f"L0")

    def get_route_numerotee(self):
        return self.route_numerotee

class Route:
    def __init__(self, classe_graph):
        self.ordre = classe_graph.get_route_numerotee()
        self.distance = 0
        self.graph = classe_graph

    def __repr__(self):
        return f"Route: {' -> '.join(str(i) for i in self.ordre)}"

    def distance_totale(self):
        return self.graph.calcul_distance_route(self.ordre)

graph_test = Graph()
graph_test.generer_lieux_aleatoires()
#graph_test.charger_graph(path)
graph_test.heuristique_voisin()

route1 = Route(graph_test)
a = route1.distance_totale()
print(route1)
print(a)
