#!/usr/bin/env python3
from __future__ import annotations

import os

import numpy as np

from dpdata.utils import open_file

from .traj import (
    kbar2evperang3,
    ry2ev,
)
from .traj import (
    length_convert as bohr2ang,
)

_QE_BLOCK_KEYWORDS = [
    "ATOMIC_SPECIES",
    "ATOMIC_POSITIONS",
    "K_POINTS",
    "ADDITIONAL_K_POINTS",
    "CELL_PARAMETERS",
    "CONSTRAINTS",
    "OCCUPATIONS",
    "ATOMIC_VELOCITIES",
    "ATOMIC_FORCES",
    "SOLVENTS",
    "HUBBARD",
]


def get_block(lines, keyword, skip=0):
    ret = []
    for idx, ii in enumerate(lines):
        if keyword in ii:
            blk_idx = idx + 1 + skip
            while len(lines[blk_idx].split()) == 0:
                blk_idx += 1
            while (
                len(lines[blk_idx].split()) != 0
                and (lines[blk_idx].split()[0] not in _QE_BLOCK_KEYWORDS)
            ) and blk_idx != len(lines):
                ret.append(lines[blk_idx])
                blk_idx += 1
            break
    return ret


def get_cell(lines):
    ret = []
    for idx, ii in enumerate(lines):
        if "ibrav" in ii:
            break
    blk = lines[idx : idx + 2]
    ibrav = int(blk[0].replace(",", "").split("=")[-1])
    if ibrav == 0:
        for iline in lines:
            if "CELL_PARAMETERS" in iline and "angstrom" not in iline.lower():
                raise RuntimeError(
                    "CELL_PARAMETERS must be written in Angstrom. Other units are not supported yet."
                )
        blk = get_block(lines, "CELL_PARAMETERS")
        for ii in blk:
            ret.append([float(jj) for jj in ii.split()[0:3]])
        ret = np.array(ret)
    elif ibrav == 1:
        a = None
        for iline in lines:
            line = iline.replace("=", " ").replace(",", "").split()
            if len(line) >= 2 and "a" == line[0]:
                # print("line = ", line)
                a = float(line[1])
            if len(line) >= 2 and "celldm(1)" == line[0]:
                a = float(line[1]) * bohr2ang
        # print("a = ", a)
        if not a:
            raise RuntimeError("parameter 'a' or 'celldm(1)' cannot be found.")
        ret = np.array([[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, a]])
    else:
        raise RuntimeError("ibrav > 1 not supported yet.")
    return ret


def get_coords(lines, cell):
    coord = []
    atom_symbol_list = []
    for iline in lines:
        if "ATOMIC_POSITIONS" in iline and (
            "angstrom" not in iline.lower() and "crystal" not in iline.lower()
        ):
            raise RuntimeError(
                "ATOMIC_POSITIONS must be written in Angstrom or crystal. Other units are not supported yet."
            )
        if "ATOMIC_POSITIONS" in iline and "angstrom" in iline.lower():
            blk = get_block(lines, "ATOMIC_POSITIONS")
            for ii in blk:
                coord.append([float(jj) for jj in ii.split()[1:4]])
                atom_symbol_list.append(ii.split()[0])
            coord = np.array(coord)
        elif "ATOMIC_POSITIONS" in iline and "crystal" in iline.lower():
            blk = get_block(lines, "ATOMIC_POSITIONS")
            for ii in blk:
                coord.append([float(jj) for jj in ii.split()[1:4]])
                atom_symbol_list.append(ii.split()[0])
            coord = np.array(coord)
            coord = np.matmul(coord, cell)
    atom_symbol_list = np.array(atom_symbol_list)
    tmp_names, symbol_idx = np.unique(atom_symbol_list, return_index=True)
    atom_types = []
    atom_numbs = []
    # preserve the atom_name order
    atom_names = atom_symbol_list[np.sort(symbol_idx, kind="stable")]
    for jj in atom_symbol_list:
        for idx, ii in enumerate(atom_names):
            if jj == ii:
                atom_types.append(idx)
    for idx in range(len(atom_names)):
        atom_numbs.append(atom_types.count(idx))
    atom_types = np.array(atom_types)

    return list(atom_names), atom_numbs, atom_types, coord


def get_energy(lines):
    energy = None
    for ii in lines:
        if "!    total energy" in ii:
            energy = ry2ev * float(ii.split("=")[1].split()[0])
    return energy


def get_force(lines, natoms):
    blk = get_block(lines, "Forces acting on atoms", skip=1)
    ret = []
    blk = blk[0 : sum(natoms)]
    for ii in blk:
        ret.append([float(jj) for jj in ii.split("=")[1].split()])
    ret = np.array(ret)
    ret *= ry2ev / bohr2ang
    return ret


def get_stress(lines):
    blk = get_block(lines, "total   stress")
    if len(blk) == 0:
        return None
    ret = []
    for ii in blk:
        ret.append([float(jj) for jj in ii.split()[3:6]])
    ret = np.array(ret)
    ret *= kbar2evperang3
    return ret


def get_frame(fname):
    if isinstance(fname, str):
        path_out = fname
        outname = os.path.basename(path_out)
        # the name of the input file is assumed to be different from the output by 'in' and 'out'
        inname = outname.replace("out", "in")
        path_in = os.path.join(os.path.dirname(path_out), inname)
    elif isinstance(fname, list) and len(fname) == 2:
        path_in = fname[0]
        path_out = fname[1]
    else:
        raise RuntimeError("invalid input")
    with open_file(path_out) as fp:
        outlines = fp.read().split("\n")
    with open_file(path_in) as fp:
        inlines = fp.read().split("\n")
    cell = get_cell(inlines)
    atom_names, natoms, types, coords = get_coords(inlines, cell)
    energy = get_energy(outlines)
    force = get_force(outlines, natoms)
    stress = get_stress(outlines)
    if stress is not None:
        stress = (stress * np.linalg.det(cell))[np.newaxis, :, :]
    return (
        atom_names,
        natoms,
        types,
        cell[np.newaxis, :, :],
        coords[np.newaxis, :, :],
        np.array(energy)[np.newaxis],
        force[np.newaxis, :, :],
        stress,
    )
