#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING

import numpy as np

from dpdata.utils import open_file

if TYPE_CHECKING:
    from dpdata.utils import FileType

lib_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(lib_path)
import warnings

import lmp


class UnwrapWarning(UserWarning):
    pass


warnings.simplefilter("once", UnwrapWarning)


def _get_block(lines, key):
    for idx in range(len(lines)):
        if ("ITEM: " + key) in lines[idx]:
            break
    idx_s = idx + 1
    for idx in range(idx_s, len(lines)):
        if ("ITEM: ") in lines[idx]:
            break
    idx_e = idx
    if idx_e == len(lines) - 1:
        idx_e += 1
    return lines[idx_s:idx_e], lines[idx_s - 1]


def get_atype(lines, type_idx_zero=False):
    blk, head = _get_block(lines, "ATOMS")
    keys = head.split()
    id_idx = keys.index("id") - 2
    tidx = keys.index("type") - 2
    atype = []
    for ii in blk:
        atype.append([int(ii.split()[id_idx]), int(ii.split()[tidx])])
    atype.sort()
    atype = np.array(atype, dtype=int)
    if type_idx_zero:
        return atype[:, 1] - 1
    else:
        return atype[:, 1]


def get_natoms(lines):
    blk, head = _get_block(lines, "NUMBER OF ATOMS")
    return int(blk[0])


def get_natomtypes(lines):
    atype = get_atype(lines)
    return max(atype)


def get_natoms_vec(lines):
    atype = get_atype(lines)
    natoms_vec = []
    natomtypes = get_natomtypes(lines)
    for ii in range(natomtypes):
        natoms_vec.append(sum(atype == ii + 1))
    assert sum(natoms_vec) == get_natoms(lines)
    return natoms_vec


def get_coordtype_and_scalefactor(keys):
    # 4 types in total,with different scaling factor
    key_pc = ["x", "y", "z"]  # plain cartesian, sf = 1
    key_uc = ["xu", "yu", "zu"]  # unwraped cartesian, sf = 1
    key_s = ["xs", "ys", "zs"]  # scaled by lattice parameter, sf = lattice parameter
    key_su = ["xsu", "ysu", "zsu"]  # scaled and unfolded,sf = lattice parameter
    lmp_coor_type = [key_pc, key_uc, key_s, key_su]
    sf = [0, 0, 1, 1]
    uw = [0, 1, 0, 1]  # unwraped or not
    for k in range(4):
        if all(i in keys for i in lmp_coor_type[k]):
            return lmp_coor_type[k], sf[k], uw[k]


def safe_get_posi(lines, cell, orig=np.zeros(3), unwrap=False):
    blk, head = _get_block(lines, "ATOMS")
    keys = head.split()
    coord_tp_and_sf = get_coordtype_and_scalefactor(keys)
    assert coord_tp_and_sf is not None, "Dump file does not contain atomic coordinates!"
    coordtype, sf, uw = coord_tp_and_sf
    id_idx = keys.index("id") - 2
    xidx = keys.index(coordtype[0]) - 2
    yidx = keys.index(coordtype[1]) - 2
    zidx = keys.index(coordtype[2]) - 2
    posis = []
    for ii in blk:
        words = ii.split()
        posis.append(
            [
                float(words[id_idx]),
                float(words[xidx]),
                float(words[yidx]),
                float(words[zidx]),
            ]
        )
    posis.sort()
    posis = np.array(posis)[:, 1:4]
    if not sf:
        posis = (posis - orig) @ np.linalg.inv(
            cell
        )  # Convert to scaled coordinates for unscaled coordinates
    if uw and unwrap:
        return (
            posis @ cell
        )  # convert scaled coordinates back to Cartesien coordinates unwrap at the periodic boundaries
    else:
        if uw and not unwrap:
            warnings.warn(
                message="Your dump file contains unwrapped coordinates, but you did not specify unwrapping (unwrap = True). The default is wrapping at periodic boundaries (unwrap = False).\n",
                category=UnwrapWarning,
            )
        return (
            (posis % 1) @ cell
        )  # Convert scaled coordinates back to Cartesien coordinates with wraping at periodic boundary conditions


def get_dumpbox(lines):
    blk, h = _get_block(lines, "BOX BOUNDS")
    bounds = np.zeros([3, 2])
    tilt = np.zeros([3])
    load_tilt = "xy xz yz" in h
    for dd in range(3):
        info = [float(jj) for jj in blk[dd].split()]
        bounds[dd][0] = info[0]
        bounds[dd][1] = info[1]
        if load_tilt:
            tilt[dd] = info[2]
    return bounds, tilt


def dumpbox2box(bounds, tilt):
    xy = tilt[0]
    xz = tilt[1]
    yz = tilt[2]
    xlo = bounds[0][0] - min(0.0, xy, xz, xy + xz)
    xhi = bounds[0][1] - max(0.0, xy, xz, xy + xz)
    ylo = bounds[1][0] - min(0.0, yz)
    yhi = bounds[1][1] - max(0.0, yz)
    zlo = bounds[2][0]
    zhi = bounds[2][1]
    info = [[xlo, xhi], [ylo, yhi], [zlo, zhi]]
    return lmp.lmpbox2box(info, tilt)


def box2dumpbox(orig, box):
    lohi, tilt = lmp.box2lmpbox(orig, box)
    xy = tilt[0]
    xz = tilt[1]
    yz = tilt[2]
    bounds = np.zeros([3, 2])
    bounds[0][0] = lohi[0][0] + min(0.0, xy, xz, xy + xz)
    bounds[0][1] = lohi[0][1] + max(0.0, xy, xz, xy + xz)
    bounds[1][0] = lohi[1][0] + min(0.0, yz)
    bounds[1][1] = lohi[1][1] + max(0.0, yz)
    bounds[2][0] = lohi[2][0]
    bounds[2][1] = lohi[2][1]
    return bounds, tilt


def load_file(fname: FileType, begin=0, step=1):
    lines = []
    buff = []
    cc = -1
    with open_file(fname) as fp:
        while True:
            line = fp.readline().rstrip("\n")
            if not line:
                if cc >= begin and (cc - begin) % step == 0:
                    lines += buff
                    buff = []
                cc += 1
                return lines
            if "ITEM: TIMESTEP" in line:
                if cc >= begin and (cc - begin) % step == 0:
                    lines += buff
                    buff = []
                cc += 1
            if cc >= begin and (cc - begin) % step == 0:
                buff.append(line)


def get_spin_keys(inputfile):
    """
    Read input file and get the keys for spin info in dump.

    Parameters
    ----------
    inputfile : str
        Path to the input file.

    Returns
    -------
    list or None
        List of spin info keys if found, None otherwise.
    """
    if inputfile is None:
        return None

    if not os.path.isfile(inputfile):
        warnings.warn(f"Input file {inputfile} not found.")
        return None

    with open(inputfile) as f:
        for line in f.readlines():
            ls = line.split()
            if (
                len(ls) > 7
                and ls[0] == "compute"
                and all(key in ls for key in ["sp", "spx", "spy", "spz"])
            ):
                compute_name = ls[1]
                return [
                    f"c_{compute_name}[{ls.index(key) - 3}]"
                    for key in ["sp", "spx", "spy", "spz"]
                ]

    return None


def get_spin(lines, spin_keys):
    """
    Get the spin info from the dump file.

    Parameters
    ----------
    lines : list
        The content of the dump file.
    spin_keys : list
        The keys for spin info in dump file.
    the spin info is stored in sp, spx, spy, spz or spin_keys, which is the spin norm and the spin vector
    1 1 0.00141160 5.64868599 0.01005602 1.54706291 0.00000000 0.00000000 1.00000000 -1.40772100 -2.03739417 -1522.64797384 -0.00397809 -0.00190426 -0.00743976
    """
    blk, head = _get_block(lines, "ATOMS")
    heads = head.split()

    if spin_keys is not None and all(i in heads for i in spin_keys):
        key = spin_keys
    else:
        return None

    try:
        idx_id = heads.index("id") - 2
        idx_sp, idx_spx, idx_spy, idx_spz = (heads.index(k) - 2 for k in key)

        norm = []
        vec = []
        atom_ids = []
        for line in blk:
            words = line.split()
            norm.append([float(words[idx_sp])])
            vec.append(
                [float(words[idx_spx]), float(words[idx_spy]), float(words[idx_spz])]
            )
            atom_ids.append(int(words[idx_id]))

        spin = np.array(norm) * np.array(vec)
        atom_ids, spin = zip(*sorted(zip(atom_ids, spin)))
        return np.array(spin)
    except (ValueError, IndexError) as e:
        warnings.warn(f"Error processing spin data: {str(e)}")
        return None


def system_data(
    lines, type_map=None, type_idx_zero=True, unwrap=False, input_file=None
):
    array_lines = split_traj(lines)
    lines = array_lines[0]
    system = {}
    system["atom_numbs"] = get_natoms_vec(lines)
    system["atom_names"] = []
    if type_map is None:
        for ii in range(len(system["atom_numbs"])):
            system["atom_names"].append("TYPE_%d" % ii)  # noqa: UP031
    else:
        assert len(type_map) >= len(system["atom_numbs"])
        for ii in range(len(system["atom_numbs"])):
            system["atom_names"].append(type_map[ii])
    bounds, tilt = get_dumpbox(lines)
    orig, cell = dumpbox2box(bounds, tilt)
    system["orig"] = np.array(orig) - np.array(orig)
    system["cells"] = [np.array(cell)]
    system["atom_types"] = get_atype(lines, type_idx_zero=type_idx_zero)
    system["coords"] = [safe_get_posi(lines, cell, np.array(orig), unwrap)]
    spin_keys = get_spin_keys(input_file)
    spin = get_spin(lines, spin_keys)
    has_spin = False
    if spin is not None:
        system["spins"] = [spin]
        has_spin = True
    for ii in range(1, len(array_lines)):
        bounds, tilt = get_dumpbox(array_lines[ii])
        orig, cell = dumpbox2box(bounds, tilt)
        system["cells"].append(cell)
        atype = get_atype(array_lines[ii], type_idx_zero=type_idx_zero)
        # map atom type; a[as[a][as[as[b]]]] = b[as[b][as^{-1}[b]]] = b[id]
        idx = np.argsort(atype, kind="stable")[
            np.argsort(np.argsort(system["atom_types"], kind="stable"), kind="stable")
        ]
        system["coords"].append(
            safe_get_posi(array_lines[ii], cell, np.array(orig), unwrap)[idx]
        )
        if has_spin:
            spin = get_spin(array_lines[ii], spin_keys)
            if spin is not None:
                system["spins"].append(spin[idx])
            else:
                warnings.warn(
                    f"Warning: spin info is not found in frame {ii}, remove spin info."
                )
                system.pop("spins")
                has_spin = False
    if has_spin:
        system["spins"] = np.array(system["spins"])
    system["cells"] = np.array(system["cells"])
    system["coords"] = np.array(system["coords"])
    return system


def split_traj(dump_lines):
    marks = []
    for idx, ii in enumerate(dump_lines):
        if "ITEM: TIMESTEP" in ii:
            marks.append(idx)
    if len(marks) == 0:
        return None
    elif len(marks) == 1:
        return [dump_lines]
    else:
        block_size = marks[1] - marks[0]
        ret = []
        for ii in marks:
            ret.append(dump_lines[ii : ii + block_size])
        # for ii in range(len(marks)-1):
        #     assert(marks[ii+1] - marks[ii] == block_size)
        return ret
    return None


if __name__ == "__main__":
    # fname = 'dump.hti'
    # lines = open(fname).read().split('\n')
    # # print(get_natoms(lines))
    # # print(get_natomtypes(lines))
    # # print(get_natoms_vec(lines))
    # posi = get_posi(lines)
    # dbox1, tilt1 = box2dumpbox(orig, box)
    # print(dbox - dbox1)
    # print(tilt - tilt1)
    # print(orig)
    # print(box)
    # np.savetxt('tmp.out', posi - orig, fmt='%.6f')
    # print(system_data(lines))
    lines = load_file("conf_unfold.dump", begin=0, step=1)
    al = split_traj(lines)
    s = system_data(lines, ["O", "H"])
    # l = np.linalg.norm(s['cells'][1],axis=1)
    # p = s['coords'][0] + l
    # np.savetxt('p',p,fmt='%1.10f')
