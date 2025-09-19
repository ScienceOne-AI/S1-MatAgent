import ase
from ase import Atom, Atoms
from ase.io import write, read
from ase.build import molecule, fcc111
from ase.geometry import find_mic
from ase.constraints import FixAtoms
import numpy as np
import pandas as pd
import random
import os

Boundary_X = 4
Boundary_Y = 4
N_Layer = 4
seed = 12

def set_global_seed(new_value):
    global seed
    if isinstance(new_value, int):
       seed = new_value
    else:
        raise ValueError("must be an integer")


# generate a H2O molecule template
def _create_h2o():
    h2o = molecule('H2O')
    h2o.rotate(180, 'x')  # rotate the molecule so that the O atom is at the bottom
    o_pos = h2o.get_positions()[0]
    h2o.translate([-o_pos[0], -o_pos[1], -o_pos[2]])
    return h2o


def _create_oh():
    oh = _create_h2o()
    oh.pop()
    return oh


# find the top sites on the surface
def _find_top_sites(atoms, tol=0.1):
    z_coords = atoms.positions[:, 2]
    return np.where(z_coords > z_coords.max() - tol)[0]

def reorder_slab(slab,element_order=[]):
    symbols = slab.get_chemical_symbols()
    positions = slab.get_positions()
    atom_data = [(symbol, pos, idx) for idx, (symbol, pos) in enumerate(zip(symbols, positions))]

    atom_data_sorted = sorted(atom_data, key=lambda x: element_order.index(x[0]))

    # Extract sorted symbols and positions
    sorted_symbols = [data[0] for data in atom_data_sorted]
    sorted_positions = [data[1] for data in atom_data_sorted]

    # Create a new Atoms object with sorted atoms
    new_slab = Atoms(symbols=sorted_symbols, positions=sorted_positions, cell=slab.get_cell(), pbc=slab.get_pbc())
    return new_slab

# find the nearest neighbor atoms on the surface
def _find_neighbors(ads_site, surface, top_indices, same_threshold=0.1):
    dist = surface.get_distances(ads_site, top_indices, mic=True)
    dist[np.where(top_indices==ads_site)[0]] = np.inf  # remove the distance to itself
    neighbor_distance = min(dist)
    neighbors = np.where(dist < neighbor_distance + same_threshold)[0]
    return top_indices[neighbors]


def _get_intermediate_points(pos1, pos2, cell, pbc, n_segments=4):
    """
    consider the periodic boundary conditions, divide the line segment between two points into n_segments segments, and return the coordinates of all segment points

    parameters:
        pos1: the coordinate of the first point (array-like, length 3)
        pos2: the coordinate of the second point (array-like, length 3)
        cell: the cell matrix (3x3 array)
        pbc: the periodic boundary conditions (list of 3 booleans)
        n_segments: the number of segments (default 4 segments, which gives 3 intermediate points)

    returns:
        numpy array, shape (n_segments+1, 3), containing the coordinates of all segment points
    """
    pos1 = np.array(pos1)
    pos2 = np.array(pos2)

    # calculate the minimum image vector considering periodic boundary conditions
    delta, _ = find_mic(pos2 - pos1, cell, pbc)

    # generate the segment ratios
    fractions = np.linspace(0, 1, n_segments + 1)

    # calculate the displacement of each segment point
    points = pos1 + fractions[:, np.newaxis] * delta
    #print(points)
    # map the point coordinates back to the cell
    if np.any(pbc):
        points = points @ np.linalg.inv(cell)
        for i in range(3):
            if pbc[i]:
                points[:, i] %= 1.0
        points = points @ cell

    return points


def prepare_slab(symbol, size=(Boundary_X, Boundary_Y, N_Layer), vacuum=10.0):  #size
    # generate the (111) surface
    slab = fcc111(symbol=symbol, size=size, vacuum=vacuum)
    slab = ase.build.sort(slab)
    return slab


def random_change_atoms(slab: Atoms, element_ratios, fix_elements_list=None, poscar_path='tmp/slab.poscar', reorder=True):
    m = len(slab) // N_Layer
    global seed
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.

    if fix_elements_list is None:
        sum_ratio = sum([p[1] for p in element_ratios])
        elements = []
        total_count = 0
        for element, ratio in element_ratios[:-1]:
            count = round(ratio * m / sum_ratio)
            total_count += count
            elements.extend([element] * count)
        elements.extend([element_ratios[-1][0]] * (m - total_count))

        random.shuffle(elements) 
    else:  # if a list of elements for each atom in a layer is given
        elements = fix_elements_list
    new_slab = slab.copy()

    # replace the atomic symbols
    new_slab.symbols = elements*N_Layer
    # Create a list of tuples with (symbol, position, index) to sort
    if reorder:
        new_slab = reorder_slab(new_slab, element_order=[p[0] for p in element_ratios])
    # fix the bottom two layers
    low=np.array([atom.z for atom in new_slab]).mean()
    c=FixAtoms(indices=[atom.index for atom in new_slab if atom.z<low])
    new_slab.set_constraint(c)
    if 'slab.symbols' in poscar_path:
        poscar_path = poscar_path.replace('slab.symbols', str(new_slab.symbols))
    os.makedirs(os.path.dirname(poscar_path), exist_ok=True)
    write(poscar_path, new_slab, vasp5=True)
    return new_slab


def absorb_h2o(slab, poscar_root_path, on_a_symbol=None, poscar_path=None):
    h_site_h2o = {}
    h2o = _create_h2o()
    top_sites = _find_top_sites(slab)
    search_sites = top_sites
    if on_a_symbol:
        symbol_top_sites=[i for i, x in enumerate(slab.get_chemical_symbols()) if x == on_a_symbol and i in top_sites]
        search_sites = np.array([random.choice(symbol_top_sites)])
    for site_idx in search_sites:
        # get the coordinates of the adsorption site
        site_pos = slab[site_idx].position

        # find the nearest neighbor atoms
        neighbors = _find_neighbors(site_idx, slab, top_sites)
        # where, find one that is adjacent to neighbors[0]
        dist = [slab.get_distances(neighbors[0], i, mic=True).item() for i in neighbors[1:]]
        neighbor_of_0 = 1 + np.argmin(dist)

        d = slab.positions[neighbors[neighbor_of_0]] - slab.positions[neighbors[0]]
        # convert the vector to fractional coordinates
        d_frac = np.linalg.solve(slab.cell.T, d.T).T  # solve the linear system to get the fractional displacement
        d_frac = d_frac - np.round(d_frac)  # limit the fractional displacement to [-0.5, 0.5)
        # convert back to cartesian coordinates to get the shortest vector
        d_min = np.dot(d_frac, slab.cell)  
        bridge = slab.positions[neighbors[0]] + d_min / 2
        d = bridge - site_pos
        # handle periodic boundary conditions
        d_frac = np.linalg.solve(slab.cell.T, d.T).T  # convert the vector to fractional coordinates
        d_frac = d_frac - np.round(d_frac)  # limit the fractional displacement to [-0.5, 0.5)

        # convert back to cartesian coordinates to get the shortest vector
        target_vec = np.dot(d_frac, slab.cell)

        # H2O
        new_slab = slab.copy()
        new_h2o = h2o.copy()
        new_h2o.rotate([0, -1, 0], target_vec)
        # translate the H2O molecule to the adsorption site, and raise the height by 2
        new_h2o.translate(site_pos + [0, 0, 2] - new_h2o[0].position)
        if new_h2o[0].position[2]>new_h2o[1].position[2]:
            delta_z = new_h2o[0].position[2]-new_h2o[1].position[2]
            new_h2o[1].position[2] += 2*delta_z
            new_h2o[2].position[2] += 2*delta_z
        # merge the H2O molecule with the slab
        new_slab.extend(new_h2o)
        # output the file
        if on_a_symbol:
            write(poscar_path, new_slab, vasp5=True)
        else:
            write(os.path.join(poscar_root_path, f'POSCAR_absorb_H2O_{site_idx}'), new_slab, vasp5=True)
        h_site_h2o[site_idx] = new_slab.positions[-1]
    return h_site_h2o


def absorb_h(slab, poscar_root_path, on_a_symbol=None, poscar_path=None):
    top_sites = _find_top_sites(slab)
    if on_a_symbol:
        symbol_top_sites=[i for i, x in enumerate(slab.get_chemical_symbols()) if x == on_a_symbol and i in top_sites]
        top_sites = [random.choice(symbol_top_sites)]
    for site_idx in top_sites:
        new_slab = slab + Atom('H', slab[site_idx].position + [0, 0, 2])  # the slab requires cartesian coordinates rather than fractional coordinates
        write(os.path.join(poscar_root_path, f'POSCAR_absorb_H_{site_idx}'), new_slab, vasp5=True)
        if on_a_symbol:
            write(poscar_path, new_slab, vasp5=True)
        else:
            write(os.path.join(poscar_root_path, f'POSCAR_absorb_H_{site_idx}'), new_slab, vasp5=True)


def absorb_oh_h(slab, possible_sites, poscar_root_path, h_site_h2o, on_a_symbol=None, poscar_path=None):
    oh = _create_oh()
    top_sites = _find_top_sites(slab)
    if on_a_symbol:
        symbol_top_sites=[i for i, x in enumerate(slab.get_chemical_symbols()) if x == on_a_symbol and i in top_sites]
        possible_sites = [random.choice(symbol_top_sites)]
    for site_idx in possible_sites:
        site_pos = slab[site_idx].position
        neighbors = _find_neighbors(site_idx, slab, top_sites)

        dist = [slab.get_distances(neighbors[0], i, mic=True).item() for i in neighbors[1:]]
        neighbor_of_0 = 1 + np.argmin(dist)

        d = slab.positions[neighbors[neighbor_of_0]] - slab.positions[neighbors[0]]

        d_frac = np.linalg.solve(slab.cell.T, d.T).T
        d_frac = d_frac - np.round(d_frac)

        d_min = np.dot(d_frac, slab.cell)  
        bridge = slab.positions[neighbors[0]] + d_min / 2
        d = bridge - site_pos

        d_frac = np.linalg.solve(slab.cell.T, d.T).T
        d_frac = d_frac - np.round(d_frac)


        target_vec = np.dot(d_frac, slab.cell)

        opposite_of_0 = 1 + np.argmax(dist)

        new_slab = slab.copy()
        new_oh = oh.copy()

        new_oh.rotate([0, -1, 0], target_vec)
        new_oh.translate(site_pos + [0, 0, 2] - new_oh[0].position)
        if new_oh[0].position[2]>new_oh[1].position[2]:
            delta_z = new_oh[0].position[2]-new_oh[1].position[2]
            new_oh[1].position[2] += 2*delta_z

        new_slab.extend(new_oh)
        h_pos = slab.positions[neighbors[opposite_of_0]] + [0, 0, 2]

        if on_a_symbol:
            write(poscar_path, new_slab + Atom('H', h_pos), vasp5=True)
        else:
            write(os.path.join(poscar_root_path, f'POSCAR_absorb_OH_H_{site_idx}'), new_slab + Atom('H', h_pos), vasp5=True)
            points = _get_intermediate_points(h_site_h2o[site_idx], h_pos, slab.cell, slab.pbc, n_segments=4)
            for i,p in enumerate(points):
                write(os.path.join(poscar_root_path, f'POSCAR_absorb_OH_H_{site_idx}_{i}'), new_slab + Atom('H', p), vasp5=True)
