# Copyright (c) 2025 Bilâl Jaiel
# This code is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License
# More details: https://creativecommons.org/licenses/by-nc/4.0/

# 2D_All_tilings_linear_system, JAIEL Bilâl, 2023.
# In this program, we use the system-solving method with parameter evaluation and the coloring algorithm.
# Here, we provide all possible solutions.

# I have added a check for connectivity here.


import numpy as np
from sympy import symbols, Eq, Matrix, solve
import re
import time
import sys
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import os

def afficher_image_png(liste_pour_affichage_png):
    
#*****************************************************************************80
#
## afficher_image_png() génère une image PNG représentant une grille de pavage colorée.
#
#  Discussion:
#
#    This function takes as input a list of matrices representing tiling pieces. Each matrix 
#    contains values indicating different cell colors. The goal is to generate a tiling image 
#    where each piece is identified by a unique number and each cell is colored based on its value. 
#    The function creates a grid where the boundaries between pieces are marked by black segments. 
#    The generated image is then saved in PNG format.
#
#    The function creates a grid of colored patches and border segments. It considers the values of 
#    adjacent cells to add border segments between cells with different values. The grid size and 
#    cell colors are adapted based on the given inputs.
#
#  Licensing:
#
#    This code is distributed under the Creative Commons Attribution-NonCommercial 4.0 International License.
#
#  Modified:
#
#    17 février 2025
#
#  Author:
#
#    Bilâl JAIEL
#
#  Input:
#
#    liste_pour_affichage_png: a list of matrices, where each matrix represents a tiling piece, 
#    and each cell contains an integer indicating the color of the cell in the piece.
#
#  Output:
#
#    No return value, but the tiling image is saved as a PNG file.  
#    The file is named 'pavage.png' or an incremented version if a file with this name already exists.
#

    # Nombre de lignes des matrices.
    n = len(liste_pour_affichage_png[0])
    # Nombre de colones des matrices.
    m = len(liste_pour_affichage_png[0][0])

    matrix = np.zeros((n,m))

    # On crée une matrice contenant toutes les pièces où chaque point d'une pièce est identifié par un numéro unique.
    k = 0
    for element in liste_pour_affichage_png :
        k = k + 1
        matrix = matrix + k*np.array(element)
 
    rows, cols = matrix.shape
    
    # Taille d'une cellule dans la grille.
    cell_size = 1.0
    
    # Création de la grille de patches colorés et de segments pour les bordures.
    patches = []
    border_segments = []
    
    for i in range(rows):
        for j in range(cols):
            value = int(matrix[i, j])

            # Liste de couleurs disponibles pour les pièces.
            colors = [(1.0, 0.0, 0.4), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0), (1.0, 0.65, 0.0),
              (0.0, 1.0, 1.0), (1.0, 0.0, 1.0), (0.5, 0.0, 0.5), (0.8, 0.4, 0.0)]
            
            if value == 0:
                color = 'white'
            else:
                color = colors[(value % 10)]
            
            # Ajout d'un rectangle coloré.
            rect = Rectangle((j, -i - 1), cell_size, cell_size, linewidth=0, edgecolor=color, facecolor=color)
            patches.append(rect)
            
            # Vérification des cellules à gauche, à droite, en bas et en haut pour ajouter des segments de bordure
            left_value = matrix[i, j - 1] if j > 0 else None
            right_value = matrix[i, j + 1] if j < cols - 1 else None
            top_value = matrix[i - 1, j] if i > 0 else None
            bottom_value = matrix[i + 1, j] if i < rows - 1 else None
            
            # Ajout de segments pour les bordures
            if left_value != value:
                border_segments.append([(j, -i - 1), (j, -i)])
            if right_value != value:
                border_segments.append([(j + 1, -i - 1), (j + 1, -i)])
            if top_value != value:
                border_segments.append([(j, -i), (j + 1, -i)])
            if bottom_value != value:
                border_segments.append([(j, -i - 1), (j + 1, -i - 1)])

    # Ajout du rectangle englobant représentant la bordure extérieure
    border_rect = Rectangle((0, -rows), cols, rows, linewidth=6, edgecolor='black', facecolor='none')
    patches.append(border_rect)

    # Affichage des patches dans une figure
    fig, ax = plt.subplots()
    for patch in patches:
        ax.add_patch(patch)

    # Affichage des segments de bordure
    for segment in border_segments:
        line = Line2D([segment[0][0], segment[1][0]], [segment[0][1], segment[1][1]], linewidth=3, color='black')
        ax.add_line(line)

    # Réglages des axes
    ax.set_xlim(0, cols)
    ax.set_ylim(-rows, 0)
    ax.set_aspect('equal', adjustable='box')

    # Suppression des axes
    ax.axis('off')

    # Enregistrement de l'image en PNG
    base_filename = '../../Output/pavage.png'
    filename = base_filename
    
    # Vérification de l'existence du fichier
    counter = 1
    while os.path.exists(filename):
        filename = f"{base_filename.split('.')[0]}_{counter}.png"
        counter += 1

    plt.savefig(filename, bbox_inches='tight', pad_inches=0, dpi=300)
    print(f"L'image a été enregistrée sous le nom '{filename}'.")

def diophantine_nd_nonnegative ( a, b ):

#*****************************************************************************80
#
## diophantine_nd_nonnegative() finds nonnegative diophantine solutions.
#
#  Discussion:
#
#     We are given a Diophantine equation 
#
#       a1 x1 + a2 x2 + ... + an * xn = b
#
#     for which the coefficients a are positive integers, and
#     the right hand side b is a nonnegative integer.
#
#     We are seeking all nonnegative integer solutions x.
#
#  Licensing:
#
#    This code is distributed under the MIT license.
#
#  Modified:
#
#    29 May 2020
#
#  Author:
#
#    John Burkardt
#
#  Input:
#
#    integer a(n): the coefficients of the Diophantine equation.
#
#    integer b: the right hand side.
#
#  Output:
#
#    integer x(k,n): solutions to the equation.
#

#
#  Initialize.
#
  n = len ( a )
  x = np.empty ( [ 0, n ] )
  j = 0
  k = 0
  r = b
  y = np.zeros ( [ 1, n ] )
#
#  Construct a vector Y that is a possible solution.
#
  while ( True ):

    r = b - sum ( a[0:j] * y[0,0:j] )
#
#  We have a partial vector Y.  Get next component.
#
    if ( j < n ):
      y[0,j] = np.floor ( r / a[j] )
      j = j + 1
#
#  We have a full vector Y.
#
    else:
#
#  Is it a solution?
#
      if ( r == 0 ):

        x = np.append ( x, y, axis = 0 )
#  
#  Find last nonzero Y entry, decrease by 1 and resume search.
#
      while ( 0 < j ):

        if ( 0 < y[0,j-1] ):
          y[0,j-1] = y[0,j-1] - 1
          break
        j = j - 1
#
#  Terminate search.
#
      if ( j == 0 ):
        break

  return x

def diophantine_nd_positive ( a, b ):

#*****************************************************************************80
#
## diophantine_nd_positive() finds positive diophantine solutions.
#
#  Discussion:
#
#     We are given a Diophantine equation 
#
#       a1 x1 + a2 x2 + ... + an * xn = b
#
#     for which the coefficients a are positive integers, and
#     the right hand side b is a nonnegative integer.
#
#     We are seeking all strictly positive integer solutions x.
#
#  Licensing:
#
#    This code is distributed under the MIT license.
#
#  Modified:
#
#    29 May 2020
#
#  Author:
#
#    John Burkardt
#
#  Input:
#
#    integer a(n): the coefficients of the Diophantine equation.
#
#    integer b: the right hand side.
#
#  Output:
#
#    integer x(k,n): solutions to the equation.
#


  beta = b - sum ( a )

  if ( beta < 0 ):
    n = len ( a )
    x = np.empty ( [ 0, n ] )
    return x

  x = diophantine_nd_nonnegative ( a, beta )
#
#  Increase every component by 1.
#
  x = x + 1
  return x

def pv_search ( parities, orders, p, c ):

#*****************************************************************************80
#
## pv_search() searches for parity violations.
#
#  Discussion:
#
#    This function considers possible tilings of a region by polyominoes.
#
#    It first determines all combinations of the polyominoes which 
#    have the same total area as the region.
#
#    Then it uses parity arguments to reject certain solutions.
#
#  Licensing:
#
#    This code is distributed under the MIT license.
#
#  Modified:
#
#    06 June 2020
#
#  Author:
#
#    Marcus Garvie,
#    John Burkardt
#
#  Input:
#
#    integer parities(nf): the parity of each polyomino.
#
#    integer orders(nf): the area each polyomino.
#
#    integer p: the parity of the region to be tiled.
#
#    integer c: the area of the region to be tiled.
#
#  Output:
#
#    integer S1(k1,nf): k1 solutions to the area equation for which
#    a trivial parity violation was found.
#
#    integer S2(k2,nf): k2 solutions to the area equation for which
#    a serious parity violation was found.
#
  import numpy as np

  nf = len ( parities )
  s1 = np.zeros ( [ 0, nf ] )
  s2 = np.zeros ( [ 0, nf ] )
#
#  Seek solutions of the area equation, { (n1, n2, ..., nF) }
#
  
  s = diophantine_nd_positive ( orders, c )
  s = s[np.all(np.isin(s, [0, 1]), axis=1), :]
  

# ns, ks = s.shape
  ns = len ( s )

  if ( ns == 0 ):
    s3 = np.concatenate((s1, s2), axis=0)
    return s, s1, s2, s3

  flags = np.zeros ( ns )
  
#
#  Remove the r zero parities.
#
  pnz = np.nonzero ( parities )
  pos_parities = parities[pnz]

#
#  Check for parity violations in each area equation solution.
# 
  for i in range ( 0, ns ):
#
#  Remove any n_i values corresponding to parities p_i = 0, 
#  i.e.,  [n_{r+1}, n_{r+2}, ..., n_F].
#
    sp = s[i,pnz]
#   sp = np.nonzeros ( ( parities > 0 ) * s[i,:] )
#
#  Flag trivial parity violations.
#   
    ps = np.sum ( pos_parities * sp )

    if ( ps < p ):
      flags[i] = 1
      continue
#
#  pos_parities*Sp = p_{r+1}*n_{r+1} + ... + p_F*n_F
#
    k = ( p + ps ) / 2  
#
#  Solve for solutions { (a_{r+1}, a_{r+2}, ..., a_F) }
#
    t = diophantine_nd_nonnegative ( pos_parities, k )
    nt = len ( t )
#   nt, kt = t.shape
#
#  There is a serious parity violation, unless at least one of the T 
#  solutions satisfies the parity condition.
# 
    flags[i] = 2
#
#  If, for any T, we have all a_k <= n_k, then S(i) does not violate parity.
#
    for j in range ( 0, nt ):

      if ( np.all ( t[j,:] <= sp ) ):
        flags[i] = 0
        break
#
#  Use the flag array to gather the trivial and serious parity violations.
#
#  S1 = rows of S with iflag = 1 (trivial parity violation).
#  S2 = rows of S with iflag = 2 (serious parity violation).
#
  i1 = np.nonzero ( flags == 1 )
  s1 = s[i1]

  i2 = np.nonzero ( flags == 2 )
  s2 = s[i2]

  s3 = np.concatenate((s1, s2), axis=0)

  return s, s1, s2, s3

def are_matrices_lines_equal(matrix1, matrix2):
#*****************************************************************************80
#
## are_matrices_lines_equal() compare deux matrices indépendamment de l'ordre des lignes.
#
#  Discussion:
#
#     Cette fonction prend en entrée deux matrices et vérifie si elles contiennent les mêmes lignes, 
#     indépendamment de l'ordre dans lequel les lignes apparaissent. La fonction trie d'abord chaque ligne de 
#     manière indépendante dans chaque matrice, puis trie les matrices entières pour ignorer l'ordre des lignes. 
#     Enfin, elle compare les deux matrices triées et retourne True si elles sont égales, sinon False.
#
#     L'objectif de cette fonction est de déterminer si deux matrices sont identiques au niveau des lignes,
#     sans tenir compte de l'ordre des lignes. Cette approche est souvent utilisée lorsque l'on veut comparer des 
#     ensembles de données sans se soucier de l'organisation des éléments dans les matrices.
#
#  Licensing:
#
#    This code is distributed under the Creative Commons Attribution-NonCommercial 4.0 International License.
#
#  Modified:
#
#    17 février 2025
#
#  Author:
#
#    Bilâl JAIEL
#
#  Input:
#
#    matrix1: une matrice de forme (n, m), où n est le nombre de lignes et m est le nombre de colonnes.
#    matrix2: une matrice de forme (n, m), qui est comparée à matrix1.
#
#  Output:
#
#    Renvoie un booléen: True si les matrices contiennent les mêmes lignes indépendamment de leur ordre, 
#    False sinon.
#

    # Triez les lignes de chaque matrice indépendamment
    sorted_matrix1 = np.sort(matrix1, axis=1)
    sorted_matrix2 = np.sort(matrix2, axis=1)
    
    # Triez les matrices triées pour ignorer l'ordre des lignes
    sorted_matrix1 = np.sort(sorted_matrix1, axis=0)
    sorted_matrix2 = np.sort(sorted_matrix2, axis=0)
    
    # Comparez les matrices triées
    return np.array_equal(sorted_matrix1, sorted_matrix2)

def set_parities(polyomino_list):

#*****************************************************************************80
#
## set_parities() calcule la parité de chaque polyomino dans une liste de polyominos.
#
#  Discussion:
#
#     Cette fonction prend en entrée une liste de polyominos et calcule la parité pour chaque polyomino.
#     Chaque polyomino est représenté par une matrice (liste de listes) où les valeurs sont 0 ou 1, et la parité 
#     est calculée en fonction de l'index des lignes et des colonnes. La parité est déterminée en utilisant une 
#     formule basée sur la somme des valeurs du polyomino multipliée par (-1) élevé à la puissance de la somme des indices.
#     Cette approche permet de calculer une sorte de "signature" pour chaque polyomino qui sera utilisée pour des 
#     applications comme la comparaison de structures ou la gestion d'ensembles de polyominos.
#
#     Le résultat est un tableau de parités pour chaque polyomino, où chaque élément correspond à la parité 
#     calculée pour un polyomino particulier dans la liste.
#
#  Licensing:
#
#    This code is distributed under the Creative Commons Attribution-NonCommercial 4.0 International License.
#
#  Modified:
#
#    17 février 2025
#
#  Author:
#
#    Bilâl JAIEL
#
#  Input:
#
#    polyomino_list: une liste de polyominos, chaque polyomino étant une matrice (liste de listes) contenant 
#    des valeurs 0 ou 1 représentant les cellules du polyomino.
#
#  Output:
#
#    Renvoie un tableau d'entiers de taille (n,), où n est le nombre de polyominos dans la liste, et chaque élément 
#    correspond à la parité calculée pour le polyomino correspondant.
#

    # Initialisation d'un tableau de parités avec des zéros pour chaque polyomino
    parities = np.zeros(len(polyomino_list), dtype=int)

    # Parcours de chaque polyomino de la liste
    for idx, polyomino in enumerate(polyomino_list):
        s = 0  # Initialisation de la somme des parités pour le polyomino actuel

        # Parcours de chaque élément du polyomino (sous-matrice)
        for i in range(len(polyomino)):
            for j in range(len(polyomino[i])):
                
                # Calcul de la contribution de chaque élément à la parité (multiplication de la valeur par (-1)^(i+j))
                s = s + polyomino[i][j] * ((-1) ** (i + j))
        
        # Stockage de la parité calculée pour ce polyomino dans le tableau des parités
        parities[idx] = s

    return parities

def set_orders(polyomino_list):
  
#*****************************************************************************80
#
## set_orders() calcule l'ordre (somme des valeurs) de chaque polyomino dans une liste de polyominos.
#
#  Discussion:
#
#     Cette fonction prend en entrée une liste de polyominos et calcule l'ordre pour chaque polyomino.
#     Chaque polyomino est représenté par une matrice (liste de listes) où les valeurs sont 0 ou 1.
#     L'ordre d'un polyomino est simplement la somme de ses éléments, c'est-à-dire la somme des valeurs dans
#     la matrice représentant le polyomino. L'ordre peut être utilisé pour classer les polyominos ou dans
#     des applications où l'on veut analyser la taille globale ou la densité des polyominos.
#
#     Le résultat est un tableau d'ordres pour chaque polyomino, où chaque élément correspond à l'ordre 
#     calculé pour un polyomino particulier dans la liste.
#
#  Licensing:
#
#    This code is distributed under the Creative Commons Attribution-NonCommercial 4.0 International License.
#
#  Modified:
#
#    17 février 2025
#
#  Author:
#
#    Bilâl JAIEL
#
#  Input:
#
#    polyomino_list: une liste de polyominos, chaque polyomino étant une matrice (liste de listes) contenant 
#    des valeurs 0 ou 1 représentant les cellules du polyomino.
#
#  Output:
#
#    Renvoie un tableau d'entiers de taille (n,), où n est le nombre de polyominos dans la liste, et chaque élément 
#    correspond à l'ordre (somme des valeurs) calculé pour le polyomino correspondant.
#

    # Initialisation d'un tableau d'ordres avec des zéros pour chaque polyomino
    parities = np.zeros(len(polyomino_list), dtype=int)

    # Parcours de chaque polyomino dans la liste
    for idx, polyomino in enumerate(polyomino_list):
        s = 0  # Initialisation de la somme des éléments du polyomino actuel (ordre)

        # Parcours de chaque élément du polyomino (sous-matrice)
        for i in range(len(polyomino)):
            for j in range(len(polyomino[i])):
                
                # Addition de la valeur de chaque élément à la somme totale s
                s = s + polyomino[i][j]
        
        # Stockage de l'ordre calculé pour ce polyomino dans le tableau des ordres
        parities[idx] = s

    # Retourne le tableau des ordres calculés
    return parities

def format_temps(temps_en_secondes):
    if temps_en_secondes < 60:
        return f"{temps_en_secondes} seconde{'s' if temps_en_secondes > 1 else ''}"
    elif temps_en_secondes < 3600:
        minutes = temps_en_secondes // 60
        secondes_restantes = temps_en_secondes % 60
        return f"{minutes} minute{'s' if minutes > 1 else ''} et {secondes_restantes} seconde{'s' if secondes_restantes > 1 else ''}"
    elif temps_en_secondes < 86400:
        heures = temps_en_secondes // 3600
        minutes_restantes = (temps_en_secondes % 3600) // 60
        return f"{heures} heure{'s' if heures > 1 else ''}, {minutes_restantes} minute{'s' if minutes_restantes > 1 else ''}"
    else:
        jours = temps_en_secondes // 86400
        heures_restantes = (temps_en_secondes % 86400) // 3600
        return f"{jours} jour{'s' if jours > 1 else ''}, {heures_restantes} heure{'s' if heures_restantes > 1 else ''}"

def index_pieces_solution(liste_bon_resultat):
    liste_index_piece_soltion = []
    n = len(liste_bon_resultat[0])
    for element in liste_bon_resultat :
        inconnues_side = []
        paramètres_side = []
        for i in range(n-1):
            if element[i]==1:
                inconnues_side.append(i)
        for j in range(len(element[-1])) :
            if int(element[-1][j]) == 1 :
                paramètres_side.append(j)
        liste_index_piece_soltion.append([inconnues_side,paramètres_side])
    return liste_index_piece_soltion

def extraction_solutions(liste_resultat) :
            solutions_extraites = []
            n = len(liste_resultat[0])
            for sub_list in liste_resultat :
                
                sortie = False

                if sub_list == [0]*(n-1)+['0'*(len(liste_resultat[-1][-1]))]:
                    sortie = True

                for i in range (n-1) :
                    if sub_list[i] not in [0,1]:
                        sortie = True
                        break

                for k in range(len(sub_list[-1][-1])) :
                        if int(sub_list[-1][-1][k]) not in [0,1] :
                            sortie = True
                            break
                
                if not sortie :
                    solutions_extraites.append(sub_list)

            return solutions_extraites

def calcul_solutions(comb,liste_2,liste_3,li) :
    liste_resultat = []
    liste_sous_resultat = []
    for j in range(len(liste_2)) :
        expression = str(li[j][1])
        variables = extraire_variables(expression)
        for element in variables :
            s = ''
            for i in range(1, len(str(element))):
                s = s + str(element)[i]
            numero = int(s)
            indice_numéro = liste_3.index(numero)
            variables[element] = int(comb[indice_numéro])
        resultat = evaluer_expression(expression, variables)
        liste_sous_resultat.append(resultat)
    liste_sous_resultat.append(comb)
    liste_resultat.append(liste_sous_resultat)
    return liste_resultat

def listes_variables_inconnues_paramètres(series,li):
    liste = []
    n = 0        
    for sublist in series:
        for matrix in sublist:
            liste.append(n)
            n = n+1

    ## variables -> liste

    liste_2 = []
    for [key,value] in li :
        sp = ''
        for j in range(1, len(str(key))):
                sp = sp + str(key)[j]
        liste_2.append(int(sp))
        
    ## inconnues -> liste_2

    liste_3 = []

    for element in liste : 
        if element not in liste_2 :
            liste_3.append(element)
    
    return [liste,liste_2,liste_3]

def extraire_variables(expression):
    # Utilise une expression régulière pour trouver tous les noms de variables dans l'expression
    pattern = r'\b[a-zA-Z_][a-zA-Z0-9_]*\b'
    variables = re.findall(pattern, expression)
    
    # Crée un dictionnaire avec des valeurs vides pour chaque variable
    variables_dict = {variable: "" for variable in variables}
    return variables_dict

def evaluer_expression(expression, variables):
    try:
        resultat = eval(expression, variables)
        return resultat
    except Exception as e:
        return f"Erreur : {e}"

def test(liste,liste_2,liste_3,li,AA):
    n = len(liste[2])
    print("2**{} = {} testes à effectuer".format(n,2**n))
    t = 0.01*(2**n)
    print("Temps d'attente estimé à : {}".format(format_temps(t)))
    input("préssez 'entrer' pour continuer...")
    start_time = time.time()  # Enregistre le temps de début
    print("'''''''''''''''''''''''''''''''''''''SOLUTIONS''''''''''''''''''''''''''''''''''''''''''")
    c = 1
    for i in range(2**n):
        comb = format(i, '0{}b'.format(n))  # Génère la représentation binaire
        # Effectuez ici l'opération souhaitée avec la variable binary_string
        liste_resultat = calcul_solutions(comb,liste_2,liste_3,li)
        ##"''''''calcul des inconnues en fonction des paramètres et les paramètres''''''")
        liste_bon_resultat = extraction_solutions(liste_resultat)
        ##"''''''selection des resultats qui conviennent (uniquement des 0 et des 1)''''''"
        if liste_bon_resultat != [] :
            liste_index_piece_solution = index_pieces_solution(liste_bon_resultat)
        else :
            liste_index_piece_solution = []
        ##''''''index correspondant''''''
       
        for element in liste_index_piece_solution :
            print()
            print("''''''solution {}''''''".format(c))
            print()
            c=c+1
            liste_pour_affichage_png = []
            if element[0] != [] :
                for i in range(len(element[0])) :
                        num_matrice = liste_2[element[0][i]]
                        liste_pour_affichage_png.append(AA[num_matrice])
            if element[1] != [] :
                    for i in range(len(element[1])) :
                        num_matrice = liste_3[element[1][i]]
                        liste_pour_affichage_png.append(AA[num_matrice])

            afficher_image_png(liste_pour_affichage_png)
            

      # Enregistre le temps de fin 544 pour le premier

    # Convertir la liste de résultats en la forme souhaitée

    # Calculer le temps réel d'exécution
    end_time = time.time()
    real_execution_time = end_time - start_time
    print(" temps d'execution :",real_execution_time)
    print()

    print("'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''")
    
    print("'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''")

def afficher(m):
    for i in range (len(m)) :
        print(m[i])

def rotation_90(m):
    n = [[0] * len(m) for _ in range(len(m[0]))]
    for i in range(len(m)):
        for j in range(len(m[0])):
            n[j][len(m) - 1 - i] = m[i][j]
    return n

def reflexion(m):
    n = [[0] * len(m[0]) for _ in range(len(m))]
    for i in range (len(m)):
        for j in range (len(m[0])):
            n[i][j] = m[i][len(m[0])-j-1]
    return n

def transformation(i, j, m, autres_pieces,aires_autres_pieces):
    """
    Génère toutes les transformations (rotations + réflexions) possibles d'un polyomino m dans une matrice i × j,
    tout en vérifiant la connectivité des espaces de 0 avec `connexity_check()`.
    """

    L = []
    B = [[0] * j for _ in range(i)]  # Matrice vide

    for _ in range(2):  # Réflexion
        m = reflexion(m)
        for _ in range(4):  # Rotations
            for a in range(len(B) - len(m) + 1):
                for b in range(len(B[0]) - len(m[0]) + 1):
                    # Copie temporaire de B pour tester l'ajout de m
                    B_temp = [row[:] for row in B]

                    # Placer m dans B_temp
                    for c in range(len(m)):
                        for d in range(len(m[0])):
                            B_temp[c + a][d + b] = m[c][d]

                    # Vérifier la connectivité après ajout
                    if connexity_check(B_temp, autres_pieces, aires_autres_pieces):
                        if B_temp not in L:
                            L.append(B_temp)

            m = rotation_90(m)  # Rotation de 90°

    return L

def resol(l, B):
    #Cette fonction prend en paramètre une liste de liste et une matrice B et renvoie une liste de liste des solutions du système

    # Coefficients des équations
    n = 0
    
    A = []
    for sublist in l:
        sublist_init = []
        for matrix in sublist:
            n = n + 1
            sublist_init.append(Matrix(matrix))
        A.append(sublist_init)
    
    x = symbols('x0:%d' % (n))
    Z = Matrix(B)
    
    # Création des équations pour chaque élément de la matrice
    equations = []
    


    for i in range(len(B)):
        for j in range(len(B[0])):
            equation = Eq(sum(A[k][s][i, j] * x[(sum(len(A[k-f]) for f in range(1, k+1)) if k != 0 else 0) + s] for k in range(len(A)) for s in range(len(A[k]))), Z[i, j])
            equations.append(equation)

    # Ajout de la contrainte
    n=0
    for sublist_matrices in A:
        equations.append(Eq(sum(x[n + s] for s in range(len(sublist_matrices))), 1))
        n = n + len(sublist_matrices)
    
    ##"'''''''''''''''''''''''''''''''''''''Equations du système'''''''''''''''''''''''''''''''''''''"

    # Résolution symbolique du système
    solution = solve(equations, x)
    
    return solution

def connexity_check(B, autres_pièces, aires_autres_pieces):
    rows = len(B)
    cols = len(B[0])
    visite = [[False] * cols for _ in range(rows)]
    minimum = min(aires_autres_pieces)

    def rec(x, y):
        """Explore récursivement tous les 0 connectés et compte leur aire."""
        if x < 0 or y < 0 or x >= rows or y >= cols or B[x][y] == 1 or visite[x][y]:
            return 0
        visite[x][y] = True
        return 1 + rec(x + 1, y) + rec(x - 1, y) + rec(x, y + 1) + rec(x, y - 1)

    count = 0
    for i in range(rows):
        for j in range(cols):
            if B[i][j] == 0 and not visite[i][j]:
                count += 1
                if count > len(autres_pièces):
                    return False  # On arrête immédiatement
                area = rec(i, j)  # Ajouter l'aire de cette zone
                if area < minimum :
                    return False

    return True  # Retourne la connectivité + l'aire totale des 0



def paving(i,j,l):
    #La fonction prend en paramètre la taille du plan et une liste de polyominos
    print("'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''")
    print("'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''")
    print("P A V A G E :")
    print()
    print("     -dimensions : {},{}".format(i,j))
    print("     -pieces : {}".format(l))
    print()
    print("'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''")

    #on règle le cas des violations de parité (le nouvelle éléments sur la V4)
    parities = set_parities(l)
    orders = set_orders(l)
    #on se limite au plan
    c = i*j
    if c%2 == 0 :
        p=0
    else :
        p=1

    s, s1, s2, s3 = pv_search(parities,orders,p,c)
        
    if are_matrices_lines_equal(s, s3):
        print('Violation total de parité, pas de pavage possible...')
        sys.exit()
    else :
        print("Pas de violation total de parité...")
    #Si ok on continue

    B = [[1] * j for _ in range(i)]
    #Series est une liste de listes de toutes les réfléxions et rotations possible de chaque polyomino
    series = []
    aires_pieces = []
    for i in range(len(l)):
        autres_pieces.append((i,sum(sum(row) for row in l[i])))
    for k in range(len(l)):
        aires_autres_pieces = aires_pieces[:]
        aires_autres_pieces.pop(k)
        autres_pieces = l[:]
        autres_pieces.pop(k)
        series.append(transformation(i, j, l[k], autres_pieces,aires_autres_pieces))
    
    print("info [à supprimer]**********\n")
    for k in range(len(series)):
        print(f"pièce n°{k+1} : {len(series[k])} dispositions selectionnées\n")
    print("fin [info à supprimer]")

    solution = resol(series,B)
    print(solution)
    # Affichage de la solution
    if solution == [] :
        print()
        print("'''''''''''''''''''''''''''''''''''''système après résolution'''''''''''''''''''''''''''''''''''''")
        print()
        print("aucune solution trouvé") 
    else :
        ##"'''''''''''''''''''''''''''''''''''''système après résolution'''''''''''''''''''''''''''''''''''''"

        li = []
        for key, value in solution.items():
            li.append([key,value])
        
        AA = []
        for sublist in series:
            for matrix in sublist:
                AA.append(matrix)
        
        ##"'''''''''''''''''''''''''''''''''''''méthode #2 : tout tester puis selectionner '''''''''''''''''''''''''''''''''''''"

        liste = listes_variables_inconnues_paramètres(series,li)
        liste_2 = liste[1]
        liste_3 = liste[2]


        test(liste,liste_2,liste_3,li,AA)




I = [[1],
     [1],
     [1]]
L = [[1,0],
     [1,0],
     [1,1]]
C = [[1]]
T = [[1,1,1],
     [0,1,0]]
U = [[1,0,1],
     [1,1,1]]

#paving(3,3,[T,U]) 
#paving(3,3,[[[0,1,0],[1,1,1],[0,1,0]],[[1,0],[1,0],[1,1]]]) 


#paving(3,2,[I,I])

paving(2,4,[L,C,[[1],[1],[1]]]) #on retrouve les resultats de la thèse
#paving(3,3,[[[1,0],[1,1]],[[1,0],[1,1]],[[1],[1],[1]]])  
#paving(5,5,[L,[[1],[1],[1]],[[1],[1],[1]],[[1],[1],[1],[1]],[[1,1],[1,1]],T,[[1],[1],[1],[0]]])
#paving(4,4,[[[1,0],[1,1]],[[1,0],[1,1]],[[1,0],[1,1]],[[1,0],[1,1]],[[1,1],[1,1]]])