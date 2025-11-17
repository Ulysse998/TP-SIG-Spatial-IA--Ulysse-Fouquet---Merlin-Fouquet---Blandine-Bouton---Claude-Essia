import numpy as np
import csv
import random
import tkinter as tk

# Constantes
LARGEUR = 800
HAUTEUR = 600
NB_LIEUX = 20


class Lieu:
    """Classe représentant un lieu avec ses coordonnées"""
    
    def __init__(self, x, y, nom):
        self.x = x
        self.y = y
        self.nom = nom
    
    def distance(self, autre_lieu):
        """Calcule la distance euclidienne entre ce lieu et un autre"""
        dx = self.x - autre_lieu.x
        dy = self.y - autre_lieu.y
        return np.sqrt(dx*dx + dy*dy)


class Graph:
    """Classe représentant un graphe de lieux à visiter"""
    
    def __init__(self):
        self.liste_lieux = []
        self.matrice_od = None
    
    def generer_lieux_aleatoires(self, nb_lieux=NB_LIEUX):
        """Génère des lieux aléatoires"""
        self.liste_lieux = []
        for i in range(nb_lieux):
            x = random.uniform(10, LARGEUR - 10)
            y = random.uniform(10, HAUTEUR - 10)
            self.liste_lieux.append(Lieu(x, y, f"Lieu_{i}"))
    
    def charger_graph(self, fichier_csv):
        """Charge les lieux depuis un fichier CSV"""
        self.liste_lieux = []
        with open(fichier_csv, 'r') as f:
            lecteur = csv.reader(f)
            next(lecteur)  # Sauter l'en-tête "x,y"
            for i, ligne in enumerate(lecteur):
                x = float(ligne[0])
                y = float(ligne[1])
                self.liste_lieux.append(Lieu(x, y, f"Lieu_{i}"))
    
    def calcul_matrice_cout_od(self):
        """Calcule la matrice des distances entre tous les lieux"""
        n = len(self.liste_lieux)
        coords = np.array([[lieu.x, lieu.y] for lieu in self.liste_lieux])
        
        # Vectorisation complète avec broadcasting
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        self.matrice_od = np.sqrt(np.sum(diff**2, axis=2))
    
    def plus_proche_voisin(self, index_lieu, lieux_restants):
        """
        Retourne l'index du plus proche voisin parmi les lieux restants
        
        Args:
            index_lieu: index du lieu de départ
            lieux_restants: set des index des lieux non encore visités
        
        Returns:
            index du plus proche voisin ou None si aucun lieu restant
        """
        if not lieux_restants:
            return None
        
        # Extraction des distances uniquement pour les lieux restants
        distances = self.matrice_od[index_lieu, list(lieux_restants)]
        index_min = np.argmin(distances)
        
        return list(lieux_restants)[index_min]
    
    def calcul_distance_route(self, route_ordre):
        """
        Calcule la distance totale d'une route
        
        Args:
            route_ordre: liste représentant l'ordre de visite des lieux
        
        Returns:
            distance totale de la route
        """
        # Vectorisation avec indexation avancée NumPy
        indices_depart = np.array(route_ordre[:-1])
        indices_arrivee = np.array(route_ordre[1:])
        
        return np.sum(self.matrice_od[indices_depart, indices_arrivee])


class Route:
    """Classe représentant une route traversant tous les lieux"""
    
    def __init__(self, graph):
        self.graph = graph
        self.ordre = [0]  # Commence toujours au lieu 0
        self.distance = float('inf')
    
    def generer_route_aleatoire(self):
        """Génère une route aléatoire visitant tous les lieux"""
        n = len(self.graph.liste_lieux)
        lieux_intermediaires = list(range(1, n))
        random.shuffle(lieux_intermediaires)
        self.ordre = [0] + lieux_intermediaires + [0]
        self.distance = self.graph.calcul_distance_route(self.ordre)
    
    def generer_route_gloutonne(self):
        """Génère une route en utilisant l'algorithme du plus proche voisin"""
        n = len(self.graph.liste_lieux)
        self.ordre = [0]
        lieux_restants = set(range(1, n))
        
        lieu_actuel = 0
        while lieux_restants:
            plus_proche = self.graph.plus_proche_voisin(lieu_actuel, lieux_restants)
            self.ordre.append(plus_proche)
            lieux_restants.remove(plus_proche)
            lieu_actuel = plus_proche
        
        self.ordre.append(0)  # Retour au point de départ
        self.distance = self.graph.calcul_distance_route(self.ordre)
    
    def calculer_distance(self):
        """Calcule et met à jour la distance de cette route"""
        self.distance = self.graph.calcul_distance_route(self.ordre)
        return self.distance


class Affichage:
    """Classe pour afficher graphiquement le graphe et les routes"""
    
    def __init__(self, graph, nom_groupe="Groupe1"):
        self.graph = graph
        self.nom_groupe = nom_groupe
        self.meilleure_route = None
        self.routes_population = []  # Pour afficher les N meilleures routes
        self.afficher_population = False
        self.matrice_pheromones = None
        
        # Création de la fenêtre
        self.root = tk.Tk()
        self.root.title(self.nom_groupe)
        
        # Canvas pour le dessin
        self.canvas = tk.Canvas(self.root, width=LARGEUR, height=HAUTEUR, bg='white')
        self.canvas.pack()
        
        # Zone de texte pour les informations
        self.texte_info = tk.Label(self.root, text="Prêt", 
                                   font=("Arial", 10), 
                                   anchor='w', justify='left')
        self.texte_info.pack(fill='x', padx=10, pady=5)
        
        # Bindings clavier
        self.root.bind('<Escape>', lambda e: self.root.destroy())
        self.root.bind('p', lambda e: self.toggle_population())
        self.root.bind('P', lambda e: self.toggle_population())
        
        self.dessiner_lieux()
    
    def dessiner_lieux(self):
        """Dessine tous les lieux du graphe"""
        rayon = 15
        for i, lieu in enumerate(self.graph.liste_lieux):
            couleur = 'red' if i == 0 else 'lightgray'
            # Cercle
            self.canvas.create_oval(lieu.x - rayon, lieu.y - rayon,
                                   lieu.x + rayon, lieu.y + rayon,
                                   fill=couleur, outline='black', width=2,
                                   tags='lieu')
            # Numéro
            self.canvas.create_text(lieu.x, lieu.y, text=str(i),
                                   font=("Arial", 10, "bold"),
                                   tags='lieu')
    
    def dessiner_route(self, route, couleur='blue', epaisseur=2, style=None):
        """Dessine une route sur le canvas"""
        if style is None:
            style = (5, 5)  # Pointillés
        
        for i in range(len(route.ordre) - 1):
            idx_depart = route.ordre[i]
            idx_arrivee = route.ordre[i + 1]
            lieu_depart = self.graph.liste_lieux[idx_depart]
            lieu_arrivee = self.graph.liste_lieux[idx_arrivee]
            
            self.canvas.create_line(lieu_depart.x, lieu_depart.y,
                                   lieu_arrivee.x, lieu_arrivee.y,
                                   fill=couleur, width=epaisseur, 
                                   dash=style, tags='route')
        
        # Afficher l'ordre de visite au-dessus des lieux
        if couleur == 'blue':
            for i, idx_lieu in enumerate(route.ordre[:-1]):
                lieu = self.graph.liste_lieux[idx_lieu]
                self.canvas.create_text(lieu.x, lieu.y - 25, 
                                       text=str(i),
                                       font=("Arial", 9),
                                       fill='blue', tags='ordre')
    
    def actualiser_affichage(self, meilleure_route, info_texte="", 
                            routes_population=None):
        """Actualise l'affichage avec la meilleure route"""
        self.meilleure_route = meilleure_route
        
        if routes_population:
            self.routes_population = routes_population
        
        # Effacer les anciennes routes
        self.canvas.delete('route')
        self.canvas.delete('ordre')
        
        # Dessiner population en gris si activé
        if self.afficher_population and self.routes_population:
            for route in self.routes_population:
                self.dessiner_route(route, couleur='lightgray', 
                                   epaisseur=1, style=(2, 2))
        
        # Dessiner la meilleure route en bleu
        if self.meilleure_route:
            self.dessiner_route(self.meilleure_route, couleur='blue', 
                               epaisseur=2, style=(5, 5))
        
        # Mettre à jour le texte d'information
        self.texte_info.config(text=info_texte)
        
        # Forcer la mise à jour
        self.root.update()
    
    def toggle_population(self):
        """Bascule l'affichage de la population"""
        self.afficher_population = not self.afficher_population
        if self.meilleure_route:
            info = self.texte_info.cget('text')
            self.actualiser_affichage(self.meilleure_route, info)
    def demarrer(self):
        """Lance la boucle principale Tkinter"""
        self.root.mainloop()


# Exemple d'utilisation
if __name__ == "__main__":
    # Créer un graphe
    graph = Graph()
    
    # Option 1: Charger depuis CSV
    graph.charger_graph(r"BDB\graph_5.csv")

    
    # Option 2: Générer aléatoirement
    #graph.generer_lieux_aleatoires(20)
    
    # Calculer la matrice de distances
    graph.calcul_matrice_cout_od()
    
    # Créer une route avec l'algorithme glouton
    route = Route(graph)
    route.generer_route_gloutonne()
    
    print(f"Route: {route.ordre}")
    print(f"Distance: {route.distance:.2f}")
    
    # Créer l'affichage
    affichage = Affichage(graph, "Groupe1")
    affichage.actualiser_affichage(route, 
        f"Distance: {route.distance:.2f} | Appuyez sur 'P' pour voir la population")
    
    # Lancer l'interface
    affichage.demarrer()