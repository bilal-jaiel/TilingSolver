# Copyright (c) 2025 Bilâl Jaiel
# This code is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License
# More details: https://creativecommons.org/licenses/by-nc/4.0/

# 2D_Fast_tiling_diophantine_equations, JAIEL Bilâl, 2024.
# In this program, we use the system-solving method with Diophantine equations and the coloring algorithm.
# Here, we seek a solution—the first one found.

# A small particularity here is that we do not solve the system.

import numpy as np
import time
import sys
import matplotlib.pyplot as plt
import os
import sympy as sp
import copy
from sympy import symbols, Matrix
from itertools import product
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D


# Cette partie est relative à la résolution d'équations diophienne et à la recherche de violation de parité

def diophantine_nd_01(a, b):
    # Author:
    # John Burkardt

    n = len(a)
    solutions = []

    for pattern in product([0, 1], repeat=n):
        if np.dot(a, pattern) == b:
            solutions.append(pattern)

    return np.array(solutions)

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

  nf = len ( parities )
  s1 = np.zeros ( [ 0, nf ] )
  s2 = np.zeros ( [ 0, nf ] )
#
#  Seek solutions of the area equation, { (n1, n2, ..., nF) }
#
  s = diophantine_nd_01 ( orders, c )
 

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


# Cette partie est relative à la recherche de violation de parité

def listsoldiph(m):
    equation = [[],1]
    for i in range (len(m)):
        if m[i] == -1 :
            equation[0].append(1)
        if m[i] != -2 and m[i] != -1  :
            equation[1]=equation[1]-m[i]
    return diophantine_nd_01(equation[0],equation[1])

def set_parities(polyomino_list):
    parities = np.zeros(len(polyomino_list), dtype=int)
    for idx, polyomino in enumerate(polyomino_list):
        s = 0
        for i in range(len(polyomino)):
            for j in range(len(polyomino[i])):
                s = s + polyomino[i][j] * ((-1) ** (i + j))
        parities[idx] = s

    return parities

def set_orders(polyomino_list):
  parities = np.zeros(len(polyomino_list), dtype=int)
  for idx, polyomino in enumerate(polyomino_list):
        s = 0
        for i in range(len(polyomino)):
            for j in range(len(polyomino[i])):
                s = s + polyomino[i][j]
        parities[idx] = s

  return parities


# Cette partie est relative à l'esthétique du code

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

def afficher_image_png(liste_pour_affichage_png):
    n_i = len(liste_pour_affichage_png[0])
    n_j = len(liste_pour_affichage_png[0][0])
    matrix = np.zeros((n_i,n_j))
    k = 0
    for element in liste_pour_affichage_png :
        k = k + 1
        matrix = matrix + k*np.array(element)
 
    rows, cols = matrix.shape
    
    # Taille d'une cellule dans la grille
    cell_size = 1.0
    
    # Création de la grille de patches colorés et de segments pour les bordures
    patches = []
    border_segments = []
    
    for i in range(rows):
        for j in range(cols):
            value = int(matrix[i, j])
            colors = [(1.0, 0.0, 0.4), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0), (1.0, 0.65, 0.0),
              (0.0, 1.0, 1.0), (1.0, 0.0, 1.0), (0.5, 0.0, 0.5), (0.8, 0.4, 0.0)]
            
            if value == 0:
                color = 'white'  # Couleur pour le "T"
            else:
                color = colors[(value % 10)]   # Couleur pour le "U"
            
            # Ajout d'un rectangle coloré
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
    print(f"L'image du pavage a été enregistrée sous le nom '{filename}'.")

def afficher_list(matrice):
    for element in matrice:
        print(element)
    print()
    print()

def afficher(m):
    for i in range (len(m)) :
        print(m[i])


# Cette partie est relative à des opérations python élémentaire

def remove_duplicates(list_of_lists):
    unique_lists = []
    for sublist in list_of_lists:
        if sublist not in unique_lists:
            unique_lists.append(sublist)
    return unique_lists

def are_matrices_lines_equal(matrix1, matrix2):
    # Triez les lignes de chaque matrice indépendamment
    sorted_matrix1 = np.sort(matrix1, axis=1)
    sorted_matrix2 = np.sort(matrix2, axis=1)
    
    # Triez les matrices triées pour ignorer l'ordre des lignes
    sorted_matrix1 = np.sort(sorted_matrix1, axis=0)
    sorted_matrix2 = np.sort(sorted_matrix2, axis=0)
    
    # Comparez les matrices triées
    return np.array_equal(sorted_matrix1, sorted_matrix2)


# Cette partie est relative à la mise en place des solutions

def index_pieces_solution(matrice):
    unique_indices = set()  # Utilisation d'un ensemble pour éliminer les doublons
    for sublist in matrice:
        indices = [i for i, value in enumerate(sublist) if value == 1]
        unique_indices.update(indices)
    return list(unique_indices)

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
        print(expression)
        variables = extraire_variables(expression)
        print(variables)
        for element in variables :
            s = ''
            for i in range(1, len(str(element))):
                s = s + str(element)[i]
            numero = int(s)
            print(li[j][1])
            print(variables)
            print(numero)
            print(liste_3)
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
    if isinstance(expression, sp.Add):
        # Récupère toutes les variables dans l'expression
        variables = expression.free_symbols
        # Crée un dictionnaire avec des valeurs vides pour chaque variable
        variables_dict = {str(variable): "" for variable in variables}
        return variables_dict
    else:
        raise TypeError("L'expression doit être de type sympy.core.add.Add")

def evaluer_expression(expression, variables):
    try:
        resultat = eval(expression, variables)
        return resultat
    except Exception as e:
        return f"Erreur : {e}"

def build_systeme(l, B):
    # Cette fonction prend en paramètre une liste de liste et une matrice B et renvoie une liste de liste qui représente le système non résolue

    # Coefficients des équations
    n = 0
    
    A = [] # Conversion de la liste de liste en matrice numpy
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
            equations.append([sum(A[0][s][i, j] * x[(sum(len(A[-1])) if 0 != 0 else 0) + s] for s in range(len(A[0])))+sum(A[k][s][i, j] * x[(sum(len(A[k-f]) for f in range(1, k+1)) if k != 0 else 0) + s] for k in range(1,len(A)) for s in range(len(A[k]))), 1])
            
    # Ajout de la contrainte
    n=0
    for sublist_matrices in A:
        equations.append([x[n]+sum(x[n + s] for s in range(1,len(sublist_matrices))) , 1])
    
        n = n + len(sublist_matrices)
    
    return equations


# Cette partie est relative à toute les rotation, réflecion et transformation des pieèces dans leur plan

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

def transformation(i,j,m) :
    # Cette fonction prend en paramètre la dimension d'un espace et une matrice d'un polyomino et renvoie toutes les réfléxions et rotations possible de ce polyomino
    L=[]
    # Création de la matrice
    B = [[0] * j for _ in range(i)]
    
    for _ in range (2) :
        m = reflexion(m)
        for _ in range (4) :
            end = True
            if len(m) <= len(B) and len(m[0]) <= len(B[0]) :
                for a in range (len(B)-len(m)+1):
                    for b in range (len(B[0])-len(m[0])+1):
                        for c in range (len(m)):
                            for d in range (len(m[0])):
                                B[c+a][d+b] = m[c][d]
                        if B not in L :
                            L.append(B)
                            B = [[0] * j for _ in range(i)]
                            end = False
                        else :
                            B = [[0] * j for _ in range(i)]
                    if end :
                        break
                if end :
                    break                    
            m = rotation_90(m)
    
    return L

    
# Cette partie est relative à la mise en place de la méthode de résolution de système avec la méthode des équations diophantienne

def assigner(m,l,og) :
    n = 0
    a=copy.deepcopy(m)
    
    for i in range (len(a[0])):
        if a[0][i] == -1 :
            for j in range(len(a)):
                if a[j][i] != -2 :
                    a[j][i] = l[0][n]
                    og[len(og)-len(a)+j][i] = l[0][n]
            n = n+1

    return a

def probleme_somme(m):
    for element in m:
        if -1 not in element :
            s = 0
            for i in range(len(element)):
                if element[i] != -2:
                    s = s + element[i]
            if s != 1 :
                return True
    return False

def matrice_complete(matrice):
    solutions = []

    def recu(m, l, og):
        while len(l) > 0:
            new_m = copy.deepcopy(assigner(m, l, og))

            v = probleme_somme(new_m)
            while v:
                l = l[1:]
                if len(l) == 0:
                    return
                new_m = copy.deepcopy(assigner(m, l, og))
                v = probleme_somme(new_m)

            if len(new_m) == 1:
                solutions.append(copy.deepcopy(og))  # Stocke la solution
                l = l[1:]  # Continue à explorer d'autres possibilités
                continue  # Passe au prochain élément de l
            
            neww_m = new_m[1:]

            recu(neww_m, listsoldiph(neww_m[0]), og)  # Explore récursivement
            l = l[1:]  # Essaye une autre combinaison

    matrice_1 = copy.deepcopy(matrice)
    recu(matrice_1, listsoldiph(matrice[0]), matrice)

    return solutions  # Retourne toutes les solutions trouvées

def genere_matrice(systeme):
    liste = []

    for element in systeme:
        sous_liste = []
        variables = extraire_variables(element[0])
        for element in variables:
            s = ''
            for i in range(1, len(str(element))):
                s += str(element)[i]
            numero = int(s)
            sous_liste.append(numero)
        liste.append(sous_liste)

    max_value = max([element for sublist in liste for element in sublist])
    p = max_value + 1

    matrice = [[-1] * p for _ in range(len(liste))]

    for i in range(len(liste)):
        for j in range(max_value + 1):
            if j not in liste[i]:
                matrice[i][j] = -2

    systme = remove_duplicates(matrice)
    print("'''''''''''''''''''''''''''''''''''''''''''''''")
    print("Nombre de variables :", len(systme[0]))
    print("Nombre d'équations :", len(systme))
    input("\npresser 'entrer' pour continuer...\n")
    print("Calculs en cours ...")
    start_time = time.time()
    
    solutions = matrice_complete(systme)  # Récupérer toutes les solutions
    
    end_time = time.time()
    real_execution_time = end_time - start_time
    print("Fin des calculs.")
    print("'''''''''''''''''''''''''''''''''''''''''''''''")
    print("Temps d'exécution :", real_execution_time, "s")
    print("'''''''''''''''''''''''''''''''''''''''''''''''")
    
    if not solutions:
        print("Aucune solution trouvée.")
        return systeme
    else:
        print(f"{len(solutions)} solutions trouvées :")
        return solutions  # Retourne toutes les solutions


# Cette fonction est la fonction principale

def paving(i,j,l):
    #La fonction prend en paramètre la taille du plan et une liste de polyominos
    print("'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''")
    print("'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''")
    print("P A V A G E :")
    print()
    print("     -dimensions : {}x{}".format(i,j))
    print("     -pieces : {}".format(l))
    print()
    print("***************************************************************************************************************")

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
    for k in range (len(l)):
        series.append(transformation(i,j,l[k]))
    
    print("\nConstruction du système en cours...\n")
    # Construction du système
    systeme = build_systeme(series,B)
    print("Fin de la construction du système.\n")
    m = genere_matrice(systeme)
    if m == systeme :
        print("pas de solution trouvé")
    else :
        for s in range(len(m)) :
            print(f"\nSOLUTION n°{s+1}: ---------------------------------------\n")
            liste = index_pieces_solution(m[s])
            liste_pieces = [item for sublist in series for item in sublist]
            AA = []
            for element in liste :
                AA.append(liste_pieces[element])
            afficher_image_png(AA)
            for element in liste :
                print()
                afficher(liste_pieces[element])


# Des exemples de pièces

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


# Ici mes différentes utilisation de ce code


#paving(3,3,[T,U])


#paving(3,3,[[[0,1,0],[1,1,1],[0,1,0]],[[1,0],[1,0],[1,1]]]) 

#paving(3,2,[I,I])

#paving(2,4,[L,C,[[1],[1],[1]]]) #on retrouve les resultats de la thèse
#paving(3,3,[[[1,0],[1,1]],[[1,0],[1,1]],[[1],[1],[1]]])  


#paving(5,5,[L,[[1],[1],[1]],[[1],[1],[1]],[[1],[1],[1],[1]],[[1,1],[1,1]],T,[[1],[1],[1],[0]]])

#paving(4,4,[[[1,0],[1,1]],[[1,0],[1,1]],[[1,0],[1,1]],[[1,0],[1,1]],[[1,1],[1,1]]])

#paving(4,4,[[[1]],[[1]],[[1,1]],[[1,1]],[[1,0,0,1],[1,1,1,1]],[[1,1],[1,1]]])
#paving(4,4,[[[1]],[[1,1]],[[1,0,0,1],[1,1,1,1]],[[1]],[[0,1,1,0],[1,1,1,1]]])


#paving(5,5,[[[1]],[[1]],[[1,1,1,1,1],[1,0,0,0,1],[1,1,0,1,1]],[[1,0,0,0,1],[1,1,1,1,1]],[[0,1,0],[0,1,0],[1,1,1]]])


#paving(5,5,[[[1,0],[1,1]],[[0,1],[0,1],[1,1]],[[1,0],[1,1]],[[0,1],[0,1],[1,1]],[[1,1,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1]],[[0,1],[0,1],[1,1]]])


#paving(5,5,[[[1]],[[1,1,1,1,1],[1,1,0,1,1]],[[1,1,1],[1,0,1]],[[1,0,0],[1,1,1]],[[1,1],[1,1],[1,1]]])

