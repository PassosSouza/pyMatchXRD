"""
XRD Analysis and Crystal Visualization Toolkit

This module provides tools for:
- Crystal structure visualization
- XRD pattern simulation
- Miller plane calculations
- Atomic position transformations
"""
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from matplotlib.gridspec import GridSpec
from itertools import permutations
import matplotlib as mpl
import spglib
#from periodictable import elements

#part for planes
from scipy.interpolate import griddata
from scipy.spatial import KDTree

# to make it prettier
from typing import List, Tuple,Dict




def xyzToabc(xyz, L, Gam):
    """Convert Cartesian coordinates (xyz) to fractional coordinates (abc)."""
    # Unpack lattice parameters and angles
    a, b, c = L  # Lattice parameters
    alpha, beta, gamma = np.radians(Gam)  # Angles in radians

    # Calculate the transformation matrix from fractional to Cartesian coordinates
    A = np.array([
        [a, b * np.cos(gamma), c * np.cos(alpha)],
        [0, b * np.sin(gamma), c * (np.cos(beta) - np.cos(alpha) * np.cos(gamma)) / np.sin(gamma)],
        [0, 0, c*np.sqrt(abs(1 - np.cos(alpha)**2 - ( (np.cos(beta) - np.cos(alpha) * np.cos(gamma)) / np.sin(gamma) )**2 ))]
    ])

    # Inverse of the transformation matrix
    A_inv = np.linalg.inv(A)

    # Cartesian coordinates (xyz) -> Fractional coordinates (abc)
    frac_coords = np.dot(A_inv, xyz)

    return frac_coords

def abcToxyz(r,L,Gam):
    """Convert fractional coordinates to Cartesian coordinates."""
    a,b,c = L
    alph,beta,gamm = np.radians(Gam)


    a_v = r[0]*a*np.array([1,0,0])
    b_vec = r[1]*b*np.array([np.cos(gamm),np.sin(gamm),0  ])
    
    c_x = np.cos(alph)
    c_y = (np.cos(beta) - np.cos(alph)*np.cos(gamm))/np.sin(gamm)
    c_z = np.sqrt(abs(1 - c_x**2 - c_y**2))

    c_vec = r[2]*c*np.array([c_x,c_y,c_z ])

    return a_v+b_vec+c_vec#<-- Mexi

def a_b_c_(L,Gamm):
    """Calculate reciprocal lattice vectors."""
    V = abs( np.dot( abcToxyz([1,0,0],L,Gamm) , np.cross(abcToxyz([0,1,0],L,Gamm) ,abcToxyz([0,0,1],L,Gamm) ) )   )+1e-9
    V2 = abs( np.dot( abcToxyz([0,1,0],L,Gamm) , np.cross(abcToxyz([0,0,1],L,Gamm) ,abcToxyz([1,0,0],L,Gamm) ) )   )+1e-9
    V3 = abs( np.dot( abcToxyz([0,0,1],L,Gamm) , np.cross(abcToxyz([1,0,0],L,Gamm) ,abcToxyz([0,1,0],L,Gamm) ) )   )+1e-9


    a_ = np.cross( abcToxyz([0,1,0],L,Gamm) ,abcToxyz([0,0,1],L,Gamm) ) / V
    b_ = np.cross( abcToxyz([0,0,1],L,Gamm) ,abcToxyz([1,0,0],L,Gamm) ) / V2
    c_ = np.cross( abcToxyz([1,0,0],L,Gamm) ,abcToxyz([0,1,0],L,Gamm) ) / V3

    return a_,b_,c_

def a_b_c_frac(L,Gamm):
    """Calculate reciprocal lattice vectors."""
    V = ( np.dot( [1,0,0] , np.cross([0,1,0] ,[0,0,1] ) )   )

    a_ = np.cross( [0,1,0] ,[0,0,1] ) / V
    b_ = np.cross( [0,0,1] , [1,0,0] ) / V
    c_ = np.cross( [1,0,0] , [0,1,0] ) / V

    return a_,b_,c_

def Ghkl(hkl,L,Gamm):
    """Calculate |G_hkl|^2 for given Miller indices."""
    h,k,l = hkl
    a_,b_,c_ = a_b_c_(L,Gamm)

    G_vector = h * a_ + k * b_ + l * c_
    return np.dot(G_vector, G_vector)

def Ghkl_frac(hkl,L,Gamm):
    """Calculate |G_hkl|^2 for given Miller indices."""
    h,k,l = hkl
    a_,b_,c_ = a_b_c_frac(L,Gamm)

    G_vector = h * a_ + k * b_ + l * c_
    return np.dot(G_vector, G_vector)


def POS(r, sym):
    """
    Evaluates symmetry operations on a given position.

    Parameters:
        r (list): A list of three coordinates [x, y, z].
        sym (list): A list of symmetry operation expressions as strings.

    Returns:
        list: A list of transformed positions based on symmetry operations.
    """
    x, y, z = r
    Pos = []
    for i in sym:
        Pos.append(eval(i))  # Apply symmetry transformation

    return Pos


# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

#Import atomic form factors factors
data = pd.read_csv(os.path.join(script_dir, "atomic_form_factors.txt"),sep=';')

def get_symmetry_operations_number(number):
    # Load the JSON file
    with open(os.path.join(script_dir, 'symm_ops.json'), 'r') as file: 
        #File I saved from pymatgen, author: Katharina Ueltzen @kaueltzen
        space_groups = json.load(file)
    for group in space_groups:
        if group['number'] == number:
            return group['symops']
    return None

def get_symmetry_operations(hermann_mauguin):
    # Load the JSON file
    with open(os.path.join(script_dir, 'symm_ops.json'), 'r') as file: 
        #File I saved from pymatgen, author: Katharina Ueltzen @kaueltzen
        space_groups = json.load(file)
    for group in space_groups:
        if group['hermann_mauguin'] == hermann_mauguin:
            return group['symops']
    return None

def INFOS_COD(name):
    """
    Extracts structural information from a file from COD.

    Parameters:
        name (str): The filename containing crystallographic data.

    Returns:
        tuple: A tuple containing:
            - unit_param (list): Lattice parameters [a, b, c].
            - unit_angles (list): Lattice angles [alpha, beta, gamma].
            - Pos_atoms (list of lists): Atomic positions grouped by element.
            - syme (list): List of symmetry operations.
            - ATOMOS (list): List of atomic species.
    """

    with open(name, 'r') as file:
        lines = file.readlines()

    unit_param = []
    unit_angles = []
    syme = []
    S = False
    Pos_atomS = []
    Atoms = False
    H_M = []
    alocs = []
    u = 0
    params = False

    for idx,i in enumerate(lines):
        #if i[0]=='#':
        #    Atoms=False
        
        #if (i == 'loop_\n') and (lines[idx+1].replace('\n','').replace(' ',"") != '_symmetry_equiv_pos_as_xyz') and (u==0) and (params):
        if idx+1<len(lines):
            if (i== 'loop_\n') and ( lines[idx+1].replace(' ','').replace('\n','')[-3:] != 'xyz' ) and (u==0) and (params):
                if 'aniso' not in lines[idx+1].replace('\n','').replace(' ','').lower():
                    if 'id' not in lines[idx+1].replace('\n','').replace(' ','').lower():
                        S = True
                        u+=1
        
        if (S) and (len(i.split(" ") )>1):
            S =  False
            Atoms = True

        if S and ((i != 'loop_\n')):
            alocs.append(i.replace(" ","").replace("\n",""))

        if Atoms and len(i.split(" ")) == 1 and i !='loop_\n':
            Atoms = False

        if (Atoms) and i!='loop_\n':
            
            pos = i.replace(" ", ",").replace('\n','')
            pos = (pos.split(','))
            

            Pos_atomS.append( [np.array(pos)[np.array(alocs) == "_atom_site_label"][0].replace("0","").replace("1","").replace("2","").replace("3","").replace("4","").replace("5","").replace("6",""),
                               float(np.array(pos)[np.array(alocs) == "_atom_site_fract_x"][0].replace("(","").replace(")","")),
                               float(np.array(pos)[np.array(alocs) == "_atom_site_fract_y"][0].replace("(","").replace(")","")),
                               float(np.array(pos)[np.array(alocs) == "_atom_site_fract_z"][0].replace("(","").replace(")",""))#,
                               #float(np.array(pos)[np.array(alocs) == "_atom_site_occupancy"][0].replace("(","").replace(")",""))
                               ]  )
        
        pos = i.replace('\n','').replace('(',"").replace(")","")
        pos = pos.replace(" ", ",")
        pos = (pos.split(','))


        if pos[0] == '_cell_length_a':
            unit_param.append(float(pos[-1]))
        if pos[0] == '_cell_length_b':
            unit_param.append(float(pos[-1]))
        if pos[0] == '_cell_length_c':
            unit_param.append(float(pos[-1]))
            params = True
        if pos[0] == '_cell_angle_alpha':
            unit_angles.append(float(pos[-1]))
        if pos[0] == '_cell_angle_beta':
            unit_angles.append(float(pos[-1]))
        if pos[0] == '_cell_angle_gamma':
            unit_angles.append(float(pos[-1]))
        if pos[0].replace(' ','') == "_symmetry_space_group_name_H-M":
            
            nn = pos[3:]
            H_M = ''
            for idd in nn:
                H_M+=idd.replace("'","") + " "
            
            if ':' in H_M:
                syme = get_symmetry_operations(H_M[:-4])
            else:
                syme = get_symmetry_operations(H_M[:-1])

    Pos_atoms = []
    #Occup = []


    poS = [ [Pos_atomS[0][1],Pos_atomS[0][2],Pos_atomS[0][3]] ]
    #occup = [[Pos_atomS[0][-1]]]
    
    ATOMOS = [Pos_atomS[0][0][:]]
    
    for i in range(1,len(Pos_atomS)):
        
        if Pos_atomS[i][0] == Pos_atomS[i-1][0]:
            poS.append( [Pos_atomS[i][1],Pos_atomS[i][2],Pos_atomS[i][3]] )
            #---
            #occup.append( [Pos_atomS[i][-1]] )
        else:
            Pos_atoms.append(poS)
            ATOMOS.append(Pos_atomS[i][0][:])
            poS = [ [Pos_atomS[i][1],Pos_atomS[i][2],Pos_atomS[i][3]] ]
            #---
            #Occup.append(occup)
            #occup = [[Pos_atomS[i][-1]]]

    Pos_atoms.append(poS)
    #--
    #Occup.append(occup)

    return np.array(unit_param),np.array(unit_angles) , Pos_atoms , syme , ATOMOS#,Occup

#==========================NUMBA===============================================================
from numba import njit

@njit
def abcToxyz_numba(r, L, Gam):
    a, b, c = (L)#np.array(L)
    alph, beta, gamm = ((Gam))*np.pi/180#np.radians(np.array(Gam))

    a_v = r[0] * a * np.array([1.0, 0.0, 0.0])
    b_vec = r[1] * b * np.array([np.cos(gamm), np.sin(gamm), 0.0])

    c_x = np.cos(alph)
    c_y = (np.cos(beta) - np.cos(alph) * np.cos(gamm)) / np.sin(gamm)
    c_z = np.sqrt(max(0.0, 1.0 - c_x**2 - c_y**2))

    c_vec = r[2] * c * np.array([c_x, c_y, c_z])

    return a_v + b_vec + c_vec

@njit
def a_b_c_numba(L, Gam):
    a_vec = abcToxyz_numba(np.array([1.0, 0.0, 0.0]), L, Gam)
    b_vec = abcToxyz_numba(np.array([0.0, 1.0, 0.0]), L, Gam)
    c_vec = abcToxyz_numba(np.array([0.0, 0.0, 1.0]), L, Gam)

    V = np.abs(np.dot(a_vec, np.cross(b_vec, c_vec))) + 1e-9
    V2 = np.abs(np.dot(b_vec, np.cross(c_vec, a_vec)))+ 1e-9
    V3 = np.abs(np.dot(c_vec, np.cross(a_vec, b_vec)))+ 1e-9

    a_ = np.cross(b_vec, c_vec) / V
    b_ = np.cross(c_vec, a_vec) / V2
    c_ = np.cross(a_vec, b_vec) / V3

    return a_, b_, c_

@njit
def Ghkl_numba(hkl, L, Gam):
    h, k, l = hkl
    a_, b_, c_ = a_b_c_numba(L, Gam)
    G_vector = h * a_ + k * b_ + l * c_
    return np.dot(G_vector, G_vector)

@njit
def generate_hkls(lamb, max_index, unit_params, unit_angles):
    max_planes = (2 * max_index + 1) ** 3
    HKLS = np.empty((max_planes, 3), dtype=np.int32)
    Counts = np.zeros(max_planes, dtype=np.int32)
    theta = np.zeros(max_planes)
    Gs = np.zeros(max_planes)
    used_angles = np.zeros(max_planes)

    counter = 0
    for h in range(-max_index, max_index + 1):
        for k in range(-max_index, max_index + 1):
            for l in range(-max_index, max_index + 1):
                if h == 0 and k == 0 and l == 0:
                    continue
                G_ = Ghkl_numba((h, k, l), unit_params, unit_angles)
                G = np.sqrt(np.abs(G_))
                sintheta = lamb * G / 2
                if sintheta >= 1:
                    continue

                angle_2theta = 2 * np.degrees(np.arcsin(sintheta))
                if angle_2theta < 5:
                    continue

                already_stored = False
                for i in range(counter):
                    if np.abs(theta[i] * 2 - angle_2theta) < 0.01:
                        Counts[i] += 1
                        if np.abs(h) + np.abs(k) + np.abs(l) < np.abs(HKLS[i][0]) + np.abs(HKLS[i][1]) + np.abs(HKLS[i][2]):
                            HKLS[i][0], HKLS[i][1], HKLS[i][2] = h, k, l
                        already_stored = True
                        break

                if not already_stored:
                    HKLS[counter][0] = h
                    HKLS[counter][1] = k
                    HKLS[counter][2] = l
                    Counts[counter] = 1
                    theta[counter] = angle_2theta / 2
                    Gs[counter] = G_
                    counter += 1

    return HKLS[:counter], Counts[:counter], theta[:counter], Gs[:counter]

#===================================================================================================


def fhlk(Z,ai,bi,ci,thetas,lamb):
    """
    used based on the cristallography book
    Calculate atomic scattering factor f(s) for an atom.
    
    f(s) = Z - 41.78214 * s^2 * sum(a_i * exp(-b_i * s^2)) + c
    
    where s = sin(theta)/lambda = |G|/2
    """
    f_all = []
    s2 = (np.sin(np.radians(thetas))/lamb )**2#Gs/(2)**2

    for i in range(len(ai)):
        f0 = 0 
        for j in range(len(ai[i])):
            f0 = f0 + ai[i][j]*np.exp( - bi[i][j] * (s2) )
        f_all.append(Z[i] - 41.78214 * (s2) * f0  + ci[i])

    return f_all

def fhlk2(Z,ai,bi,ci,thetas,lamb):
    """
    Calculate atomic scattering factor f(s) for an atom.
    
    f(s) = sum(a_i * exp(-b_i * s^2)) + c
    
    where s = sin(theta)/lambda = |G|/2
    
    Not based on the book
    """
    f_all = []
    s2 = (np.sin(np.radians(thetas))/lamb )**2 #Gs/(2)**2

    for i in range(len(ai)):
        f0 = 0 
        for j in range(len(ai[i])):
            f0 = f0 + ai[i][j]*np.exp( - bi[i][j] * (s2) )
        f_all.append( f0  + ci[i])

    return f_all    

def Lp(theta):
    """Calculate Lorentz polarization correction factor."""
    theta = np.radians(theta)

    return (1 + ( np.cos(2*theta) )**2 )/( np.sin(theta)**2 * np.cos(theta) )


def FhklBTphkl(hkls, thetas, Pos, fs, lamb, L, Gamm, BT, counts):
    """
    Calculate structure factors F_hkl for all hkl planes.
    
    Args:
        hkls: Array of Miller indices (h,k,l)
        thetas: Scattering angles for each hkl
        Pos: List of atomic positions (fractional coordinates)
        Z: Atomic numbers
        ai, bi, ci: Atomic scattering parameters
        lamb: Wavelength
        L: Unit cell parameters [a, b, c]
        Gamm: Unit cell angles [alpha, beta, gamma]
        BT: Debye-Waller factors for each atom
        counts: Multiplicity factors
        
    Returns:
        Complex array of structure factors
    """
    F_all = [] #np.zeros(len(hkls), dtype=complex)
    a_,b_,c_ = a_b_c_(L,Gamm)


    for k in range(len(hkls)):
        F = 0 + 0j

        G_vector = hkls[k][0]*a_ + hkls[k][1]*b_ + hkls[k][2]*c_

        s2 = (np.sin(np.radians(thetas[k]))/lamb )**2 #Gs[k]/(2)**2
        
        for i in range(len(Pos)):
            Debye = (np.exp(-BT[i]* ( s2 ) ) )

            for j in range(len(Pos[i])):

                F += fs[i][k] * np.exp( 2j*np.pi  * ( np.dot( (Pos[i][j]) , G_vector ) ) ) #* Debye
                
        F_all.append( (F) ) 

    return np.array(F_all) 

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

#Import atomic form factors factors
data = pd.read_csv(os.path.join(script_dir, "atomic_form_factors.txt"),sep=';')
#data = pd.read_csv(os.path.join(script_dir, "book_atomic_form_factors.txt"),sep=';')
Elements = data['Element']
a_1 = data['a1'].values
a_2 = data['a2'].values
a_3 = data['a3'].values
a_4 = data['a4'].values

b_1 = data['b1'].values
b_2 = data['b2'].values
b_3 = data['b3'].values
b_4 = data['b4'].values

c_1 = data['c'].values

Z0 = data['Z'].values

def associate_atomic_factor(Element):
    a = []
    b = []
    c = []
    Z = []
    for i in Element:
        g = ((Elements==i))

        a_ = []
        a_.append(a_1[g][0])
        a_.append(a_2[g][0])
        a_.append(a_3[g][0])
        a_.append(a_4[g][0])
        a.append(a_)

        b_ = []
        b_.append(b_1[g][0])
        b_.append(b_2[g][0])
        b_.append(b_3[g][0])
        b_.append(b_4[g][0])
        b.append(b_)

        c.append(c_1[g][0])

        Z.append(Z0[g][0])

    return a,b,c,Z


def simulate_xrd(lamb, max_index, unit_params, unit_angles , positions, 
                Base_Atoms):
    """
    Simulate XRD pattern for a crystal structure.
    
    Args:
        lamb: Wavelength in angstroms
        max_index: Maximum Miller index to consider
        unit_params: [a, b, c] in angstroms
        unit_angles: [alpha, beta, gamma] in degrees
        positions: List of atomic positions (fractional coordinates)
        Zs: Atomic numbers
        a_coeffs, b_coeffs, c_coeffs: Atomic scattering parameters
        B_factors: Debye-Waller factors for each atom
        
    Returns:
        Dictionary with XRD pattern data:
        {
            'hkls': Miller indices,
            'two_thetas': Diffraction angles,
            'intensities': Calculated intensities,
            'Gs': reciprocal distances,
            'multiplicities': Multiplicity factors
        }
    """

    # Calculate atomic factors
    a_coeffs,b_coeffs,c_coeffs,Zs = associate_atomic_factor(Base_Atoms)

    #import Debyw Waller Factors
    data = pd.read_csv(os.path.join(script_dir, "Debye_Waller_factor.txt"),sep=';')#pd.read_csv('Debye_Waller_factor.txt',sep = ';')
    ElementsB = data['Element']
    BT = data['B (A)'].values

    B_factors = [BT[ElementsB == i][0] for i in Base_Atoms]
    # Generate possible hkl planes
    hkls, counts, thetas, Gs = generate_hkls(lamb, max_index, (unit_params), (unit_angles)) #<--- edited here
    fs = fhlk2(Zs,a_coeffs,b_coeffs,c_coeffs,thetas,lamb)
    
    # Calculate structure factors
    F_hkls = FhklBTphkl(hkls , thetas , positions , fs,lamb , unit_params, unit_angles,B_factors, counts  )
    
    # Calculate intensities
    intensities = np.abs(F_hkls)**2*Lp(thetas)*counts 
    
    # Normalize to maximum intensity of 1
    if len(intensities) == 0:
        return {
        'hkls': [],#[mask],
        'two_thetas': [],#[mask],
        'FsB': [],#[mask],
        'intensities': [],#[mask],
        'G': [],#[mask]), #d = 1 / Gs
        'multiplicities': [],#[mask],
        'Lp':[],#[mask]),
        'fhkls':[],
        'Zs':[]
        }

    intensities = 100 * intensities / (np.max(intensities)+1e-10 )

    
    return {
        'hkls': hkls[intensities >=2],#[mask],
        'two_thetas': 2*np.array(thetas)[intensities >=2],#[mask],
        'FsB':F_hkls[intensities >=2],#[mask],
        'intensities': intensities[intensities >=2],#[mask],
        'G': np.sqrt(Gs[intensities >=2]),#[mask]), #d = 1 / Gs
        'multiplicities': counts[intensities >=2],#[mask],
        'Lp':Lp(thetas[intensities >=2]),#[mask]),
        'fhkls':fs,
        'Zs':Zs
    }

def Intensid_xrd(lamb, hkls,thetas,counts, unit_params, unit_angles , positions, 
                Base_Atoms):
    """
    Simulate XRD pattern for a crystal structure.
    
    Args:
        lamb: Wavelength in angstroms
        max_index: Maximum Miller index to consider
        unit_params: [a, b, c] in angstroms
        unit_angles: [alpha, beta, gamma] in degrees
        positions: List of atomic positions (fractional coordinates)
        Zs: Atomic numbers
        a_coeffs, b_coeffs, c_coeffs: Atomic scattering parameters
        B_factors: Debye-Waller factors for each atom
        
    Returns:
        Dictionary with XRD pattern data:
        {
            'hkls': Miller indices,
            'two_thetas': Diffraction angles,
            'intensities': Calculated intensities,
            'Gs': reciprocal distances,
            'multiplicities': Multiplicity factors
        }
    """

    # Calculate atomic factors
    a_coeffs,b_coeffs,c_coeffs,Zs = associate_atomic_factor(Base_Atoms)

    #import Debyw Waller Factors
    data = pd.read_csv(os.path.join(script_dir, "Debye_Waller_factor.txt"),sep=';')#pd.read_csv('Debye_Waller_factor.txt',sep = ';')
    ElementsB = data['Element']
    BT = data['B (A)'].values

    B_factors = [BT[ElementsB == i][0] for i in Base_Atoms]
    # Generate possible hkl planes
    fs = fhlk2(Zs,a_coeffs,b_coeffs,c_coeffs,thetas,lamb)
    
    # Calculate structure factors
    F_hkls = FhklBTphkl(hkls , thetas , positions , fs,lamb , unit_params, unit_angles,B_factors, counts  )
    
    # Calculate intensities
    intensities = np.abs(F_hkls)**2*Lp(thetas)*counts 
    
    # Normalize to maximum intensity of 1
    if len(intensities) == 0 or np.max(intensities)==0:
        return {
        'hkls': [],#[mask],
        'two_thetas': [],#[mask],
        'FsB': [],#[mask],
        'intensities': [],#[mask],
        'multiplicities': [],#[mask],
        'Lp':[],#[mask]),
        'fhkls':[],
        'Zs':[]
        }

    intensities = 100 * intensities / (np.max(intensities)+1e-10)

    
    return {
        'hkls': hkls[intensities >=2],#[mask],
        'two_thetas': 2*thetas[intensities >=2],#[mask],
        'FsB':F_hkls[intensities >=2],#[mask],
        'intensities': intensities[intensities >=2],#[mask],
        'multiplicities': counts[intensities >=2],#[mask],
        'Lp':Lp(thetas[intensities >=2]),#[mask]),
        'fhkls':fs,
        'Zs':Zs
    }

def Normalize(data, max=1):
    # Check if it's a single list (not nested)
    if all(isinstance(i, (int, float)) for i in data):
        # Process the single list
        return [
            x - max if x > max else (x + 1 if x < 0 else x)
            for x in data
        ]
    else:
        # Process each sublist in the nested list
        return [
            [x - max if x > max else (x + 1 if x < 0 else x) for x in sublist]
            for sublist in data
        ]


def find_atoms_unit_cell(Pos_atoms, unit_params, unit_angles, Symmetry, units=[1, 1, 1], hex=False):
    """
    Expands the unit cell to multiple repetitions along each axis.

    Parameters:
        Pos_atoms (list): List of atomic positions in fractional coordinates.
        unit_params (list): Lattice parameters [a, b, c].
        unit_angles (list): Lattice angles [alpha, beta, gamma].
        Symmetry (list): List of symmetry operations.
        units (list, optional): Number of unit cells along [x, y, z]. Default is [1,1,1].
        hex (bool, optional): If True, applies a specific hexagonal expansion.

    Returns:
        tuple: (POS_unit, POS_tot) - Atomic positions expanded within the unit cell and full expanded positions.
    """

    # Step 1: Apply symmetry operations and normalize atomic positions
    POS_primit = [
        list({tuple(np.round(h, 3)) for pos in atom_list for h in (POS(pos, Symmetry))})
        for atom_list in Pos_atoms
    ]

    ppos_primit = [
        [ h
            for pos in atom_list
            for h in POS(pos, Symmetry)
        ]  for atom_list in Pos_atoms    ]
    
    primit = []
    for i in range(len(ppos_primit)):
        pp = []
        verif = []
        for j in range(len(ppos_primit[i])):
            vec = Normalize(ppos_primit[i][j])
            if 0 <= vec[0] < 1 and 0 <= vec[1] < 1 and 0 <= vec[2] < 1:
                cart = tuple(np.round(abcToxyz(vec, unit_params, unit_angles) ,2))
                if cart not in verif:
                    verif.append(cart)
                    pp.append(abcToxyz(vec, unit_params, unit_angles))
        primit.append(pp)


    # Step 2: Define unit cell boundaries and displacement vectors
    if hex:
        a, b, c = units
        vertices = np.array([
            [ a/2, -b/2 * np.sqrt(3), 0], [ a, 0, 0], [ a/2, b/2 * np.sqrt(3), 0],
            [-a/2, b/2 * np.sqrt(3), 0], [-a, 0, 0], [-a/2, -b/2 * np.sqrt(3), 0],
            [ a/2, -b/2 * np.sqrt(3), c], [ a, 0, c], [ a/2, b/2 * np.sqrt(3), c],
            [-a/2, b/2 * np.sqrt(3), c], [-a, 0, c], [-a/2, -b/2 * np.sqrt(3), c]
        ])
        x_min, y_min, z_min = np.min(vertices, axis=0)
        x0_min,y0_min,z0_min = x_min,y_min,z_min
        x_max, y_max, z_max = np.max(vertices, axis=0)
        x0_max,y0_max,z0_max = x_max,y_max,z_max
        d = np.array([[dx, dy, dz] for dx in range(-a, a+1) for dy in range(-b, b+1) for dz in range(-c, c+1)])
    else:
        x_min, y_min, z_min, x_max, y_max, z_max = 0, 0, 0, units[0], units[1], units[2]
        x0_min, y0_min, z0_min = 0, 0, 0
        x0_max, y0_max, z0_max = 1, 1, 1
        d = np.array([[dx, dy, dz] for dx in range(x_min,units[0]+1) for dy in range(y_min,units[1]+1) for dz in range(z_min,units[2]+1)])

    # Step 3: Expand atomic positions across unit cells
    POS_tot, POS_unit = [], []
    
    for i in range(len(POS_primit)):
        # Use sets to store unique atomic positions
        tot_p_set = set()
        unit_p_set = set()

        for v in d:
            for j in range(len(POS_primit[i])):
                r_r = np.array(POS_primit[i][j]) + np.array(v)  # Shift atomic position

                if x_min <= r_r[0] <= x_max and y_min <= r_r[1] <= y_max and z_min <= r_r[2] <= z_max:
                    aa = tuple(np.round(abcToxyz(r_r, unit_params, unit_angles), 3))  # Convert to tuple

                    tot_p_set.add(aa)  # Set automatically removes duplicates

                    if x0_min <= r_r[0] <= x0_max and y0_min <= r_r[1] <= y0_max and z0_min <= r_r[2] <= z0_max:
                        unit_p_set.add(aa)

        # Convert sets back to lists
        POS_tot.append(list(tot_p_set))
        POS_unit.append(list(unit_p_set))

    return primit,POS_unit,POS_tot



#Para Plots
def Plot_Planes(ax,vector,L,Gamm, num_planes = 1 , xs = [-1,5] , ys =[-1,5] , zs =[-0.5,21] ,color = 'red',alpha = 0.3):
    """
    Plots a plane perpendicular to a given vector in 3D space.

    Parameters:
        vector (list or tuple): The normal vector to the plane (a, b, c).
        plane_size (float): The size of the plane grid to be plotted.
    """

    x_min,y_min,z_min = -num_planes *max(xs), -num_planes *max(ys), -num_planes *max(zs)
    x_max,y_max,z_max = num_planes * max(xs), num_planes *max(ys), max(zs)


    N = 400

    a, b, c = vector 
    points = []  

    if a!=0:
        points.append(abcToxyz([1/a,0,0],L,Gamm))
    if b!=0:
        points.append(abcToxyz([0,1/b,0],L,Gamm))
    if c!=0:
        points.append(abcToxyz([0,0,1/c],L,Gamm))

    if len(points) == 1:
        # Single point case: plot a plane passing through the point

        if b!=0 and c!=0:
            x,y,z = points[0]
            p1 = abcToxyz([1,0,0],L,Gamm) #-points[0]
            d = -x*p1[0] - y*p1[1] - z*p1[2]

            xi, yi = np.meshgrid(np.linspace(x_min,x_max, N),
                            np.linspace(y_min, y_max, N))
            zi = (-d - x*xi - y*yi) / z

        elif a!=0 and c!=0:
            x,y,z = points[0]
            p1 = abcToxyz([0,1,0],L,Gamm) #-points[0]
            d = -x*p1[0] - y*p1[1] - z*p1[2]

            yi, zi = np.meshgrid(np.linspace(y_min,y_max, N),
                            np.linspace(z_min, z_max, N))
            xi = (-d - y*yi - z*zi) / x

        elif a!=0 and b!=0 :
            x,y,z = points[0]
            p1 = abcToxyz([0,0,1],L,Gamm) #-points[0]
            d = -x*p1[0] - y*p1[1] - z*p1[2]

            xi, zi = np.meshgrid(np.linspace(x_min,x_max, N),
                            np.linspace(z_min, z_max, N))
            yi = (-d - x*xi - z*zi) / y

        #--------------
        elif a==0 and b==0:
            p1 = abcToxyz([1,0,0],L,Gamm) #- points[0]
            p2 = abcToxyz([0,1,0],L,Gamm) #- points[0]
            x,y,z = np.cross(p1,p2)

            d = -x*points[0][0] - y*points[0][1] - z*points[0][2]

            xi, yi = np.meshgrid(np.linspace(x_min,x_max, N),
                            np.linspace(y_min, y_max, N))
            zi = (-d - x*xi - y*yi) / z
            
        elif c==0 and a==0:
            p1 = abcToxyz([1,0,0],L,Gamm) #- points[0]
            p2 = abcToxyz([0,0,1],L,Gamm) #- points[0]
            x,y,z = np.cross(p1,p2)

            d = -x*points[0][0] - y*points[0][1] - z*points[0][2]

            xi, zi = np.meshgrid(np.linspace(x_min,x_max, N),
                            np.linspace(z_min, z_max, N))
            yi = (-d - x*xi - z*zi) / y
        elif b==0 and c==0:
            p1 = abcToxyz([0,1,0],L,Gamm) #- points[0]
            p2 = abcToxyz([0,0,1],L,Gamm) #- points[0]
            x,y,z = np.cross(p1,p2)

            d = -x*points[0][0] - y*points[0][1] - z*points[0][2]

            zi, yi = np.meshgrid(np.linspace(z_min,z_max, N),
                            np.linspace(y_min, y_max, N))
            xi = (-d - z*zi - y*yi) / x
        

    elif len(points) == 2:
        # Two points case: plot a line between the points
        p1 = points[0]
        p2 = points[1]
        
        v1 = p1-p2
        
        if c==0:
            v2 = abcToxyz([0,0,1],L,Gamm) - p2
            normal_vector = np.cross(v1,v2)
            a0, b0, c0 = normal_vector

            d = +a0*p1[0] + b0*p1[1] + c0*p1[2]

            # Generate points on the plane
            xi, zi = np.meshgrid(np.linspace(x_min,x_max, N),
                            np.linspace(z_min, z_max, N))
            yi = (d - a0*xi - c0*zi) / b0
            yi = np.ma.masked_where(yi < y_min, yi)
            yi = np.ma.masked_where(yi > y_max, yi)
            x,y,z = normal_vector
        elif a==0:
            v2 = abcToxyz([1,0,0],L,Gamm)-p2
            normal_vector = np.cross(v1,v2)
            a0, b0, c0 = normal_vector

            d = -a0*p1[0] - b0*p1[1] - c0*p1[2]

            # Generate points on the plane
            xi, yi = np.meshgrid(np.linspace(x_min,x_max, N),
                            np.linspace(y_min, y_max, N))
            zi = (-d - a0*xi - b0*yi) / c0
            zi = np.ma.masked_where(zi < z_min, zi)
            zi = np.ma.masked_where(zi > z_max, zi)
            x,y,z = normal_vector
        elif b==0:
            v2 = abcToxyz([0,1,0],L,Gamm)-p2
            normal_vector = np.cross(v1,v2)
            a0, b0, c0 = normal_vector

            d = -a0*p1[0] - b0*p1[1] - c0*p1[2]


            # Generate points on the plane
            xi, yi = np.meshgrid(np.linspace(x_min,x_max, N),
                            np.linspace(y_min, y_max, N))
            zi = (-d - a0*xi - b0*yi) / c0
            zi = np.ma.masked_where(zi < z_min, zi)
            zi = np.ma.masked_where(zi > z_max, zi)
            x,y,z = normal_vector
    else:
        # Three or more points case: fit a plane using least squares
        X = points[0]
        Y = points[1]
        Z = points[2]

        v1= X-Y
        v2 = Z-Y
 
        normal_vector = np.cross(v1,v2)
        x,y,z = normal_vector

        # Define a point on the plane (the plane's intercept along the Z-axis, for instance)
        d = np.dot(normal_vector,Y)


        # Create a grid of x, y values
        x_vals = np.linspace(x_min, x_max, N)
        y_vals = np.linspace(y_min, y_max, N)
        xi, yi = np.meshgrid(x_vals, y_vals)

        # Calculate Z values based on the plane equation
        zi = (d - normal_vector[0] * xi - normal_vector[1] * yi )/normal_vector[2]
        zi = np.ma.masked_where(zi < z_min, zi)
        zi = np.ma.masked_where(zi > z_max, zi)

    norm = np.sqrt(x**2 + y**2 + z**2)* np.sqrt(Ghkl(vector,L,Gamm))
    spacing = np.array([np.abs(x), np.abs(y), np.abs(z)]) / norm #<-- later review the abs()

    Xs,Ys,Zs = [], [], []
    #rs = []
    for i in range(num_planes):
        # Calculate translation
        translation = i * spacing
        
        # Apply translation and mask out-of-bounds values
        x_ = xi + translation[0]
        y_ = yi + translation[1]
        z_ = zi + translation[2]

        # Create masks for each dimension
        x_mask = (x_ >= min(xs)) & (x_ <= max(xs))
        y_mask = (y_ >= min(ys)) & (y_ <= max(ys))
        z_mask = (z_ >= min(zs)) & (z_ <= max(zs))

        # Combine masks (all conditions must be True)
        combined_mask = x_mask & y_mask & z_mask

        # Apply masking to all arrays
        x_masked = np.ma.masked_where(~combined_mask, x_)
        y_masked = np.ma.masked_where(~combined_mask, y_)
        z_masked = np.ma.masked_where(~combined_mask, z_)


        ax.plot_surface(x_masked,y_masked,z_masked , facecolor=color ,alpha = alpha, edgecolor = 'none')#color,linewidth=2)   
        Xs.append(x_masked)
        Ys.append(y_masked)
        Zs.append(z_masked)

    return Xs,Ys,Zs



def plot_points_on_surface(ax,points, Xs,Ys,Zs,Z0 ,colors = 'red',threshold = 0.4):
    """
    Plots points on a surface.

    Args:
        points: A list of points to plot.
        surface_points: A list of points defining the surface.
    """
    for i in range(len(Xs)):
        X = Xs[i]#+=v[0]
        Y= Ys[i]#+=v[1]
        Z= Zs[i]#+=v[2]
        # Create a KDTree for efficient nearest neighbor search
        surface_points = np.vstack((X.flatten(), Y.flatten(), Z.flatten())).T
        tree = KDTree(surface_points)

        # Find the nearest point on the surface for each point in the list
        distances, indices = tree.query(points, k=1)

        # Filter points that are close enough to the surface
        #threshold = 0.4  # Adjust the threshold as needed
        filtered_points = [points[l] for l in range(len(points)) if distances[l] < threshold]

        
        for j in range(len(filtered_points)):
            ax.scatter(filtered_points[j][0] ,filtered_points[j][1],filtered_points[j][2] , color=colors,s=Z0)

def UNIT_CELL_PLOT(ax, L, Gamm, units=[1, 1, 1], xyz=True):
    """
    Plots a 3D visualization of multiple unit cells based on the given dimensions.

    Parameters:
        ax: Matplotlib 3D axis object to plot on.
        L (list): Lattice parameters [a, b, c].
        Gamm (list): Lattice angles [alpha, beta, gamma].
        units (list, optional): Number of unit cells along [x, y, z]. Default is [1, 1, 1].
        xyz (bool, optional): If True, converts fractional coordinates to Cartesian.
    """
    # Define the vertices of a unit cell at the origin
    vertices = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
    ])

    # Define the edges connecting the vertices
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
        [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
        [0, 4], [1, 5], [2, 6], [3, 7]   # Side edges
    ]

    # Loop through the number of unit cells in each direction
    for dx in range(units[0]):
        for dy in range(units[1]):
            for dz in range(units[2]):
                shift = np.array([dx, dy, dz])  # Translation vector
                
                for edge in edges:
                    p1 = vertices[edge[0]] + shift
                    p2 = vertices[edge[1]] + shift
                    
                    if xyz:
                        p1, p2 = abcToxyz(p1, L, Gamm), abcToxyz(p2, L, Gamm)
                    
                    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color='black')


def UNIT_CELL_PLOT_HEXAGON(ax, L, Gamm, xyz=True):
    """
    Plots a centered hexagonal unit cell (hexagonal prism) in 3D.

    Parameters:
    - ax: Matplotlib 3D axis
    - L: Lengths of the unit cell (a, b, c)
    - Gamm: Angles of the unit cell (alpha, beta, gamma) in degrees
    - xyz: If True, converts lattice points using abcToxyz function; otherwise uses Cartesian coordinates.
    """
    a, b, c = [1,1,1]  # Lattice parameters
    # Define the vertices of a centered hexagonal prism
    vertices = np.array([
            # Bottom face vertices
            [ a/2, -b/2 * np.sqrt(3), 0],
            [ a, 0, 0],
            [ a/2, b/2 * np.sqrt(3), 0],
            [-a/2, b/2 * np.sqrt(3), 0],
            [-a, 0, 0],
            [-a/2, -b/2 * np.sqrt(3), 0],
            
            # Top face vertices
            [ a/2, -b/2 * np.sqrt(3), c],
            [ a, 0, c],
            [ a/2, b/2 * np.sqrt(3), c],
            [-a/2, b/2 * np.sqrt(3), c],
            [-a, 0, c],
            [-a/2, -b/2 * np.sqrt(3), c]
        ])

        # Define edges connecting the vertices for a hexagonal prism
    edges = [
            # Bottom face edges
            [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0],
            
            # Top face edges
            [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 6],
            
            # Side edges connecting top and bottom faces
            [0, 6], [1, 7], [2, 8], [3, 9], [4, 10], [5, 11]
    ]


    # Plot the edges of the hexagon
    for edge in edges:
        if xyz:
            # Convert lattice points if necessary
            p1 = (vertices[edge[0]])
            p2 = (vertices[edge[1]])
            p1, p2 = abcToxyz(p1, L, Gamm), abcToxyz(p2, L, Gamm)
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color='gray')
        else:
            p1, p2 = vertices[edge[0]], vertices[edge[1]]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color='gray')