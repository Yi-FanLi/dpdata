from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from dpdata.format import Format
from dpdata.orca.output import read_orca_sp_output
from dpdata.unit import EnergyConversion, ForceConversion

if TYPE_CHECKING:
    from dpdata.utils import FileType

energy_convert = EnergyConversion("hartree", "eV").value()
force_convert = ForceConversion("hartree/bohr", "eV/angstrom").value()


@Format.register("orca/spout")
class ORCASPOutFormat(Format):
    """ORCA single point energy output.

    Note that both the energy and the gradient should be
    printed into the output file.
    """

    def from_labeled_system(self, file_name: FileType, **kwargs) -> dict:
        """Read from ORCA single point energy output.

        Parameters
        ----------
        file_name : FileType
            file name
        **kwargs
            keyword arguments

        Returns
        -------
        dict
            system data
        """
        symbols, coord, energy, forces = read_orca_sp_output(file_name)

        atom_names, atom_types, atom_numbs = np.unique(
            symbols, return_inverse=True, return_counts=True
        )
        natoms = coord.shape[0]

        return {
            "atom_types": atom_types,
            "atom_names": list(atom_names),
            "atom_numbs": list(atom_numbs),
            "coords": coord.reshape((1, natoms, 3)),
            "energies": np.array([energy * energy_convert]),
            "forces": (forces * force_convert).reshape((1, natoms, 3)),
            "cells": np.zeros((1, 3, 3)),
            "orig": np.zeros(3),
            "nopbc": True,
        }

@Format.register("orca/inp")
class ORCAInpFormat(Format):
    """ORCA input file."""

    def to_system(self, data: dict, file_name: FileType, **kwargs):
        """Generate ORCA input file.

        Parameters
        ----------
        data : dict
            system data
        file_name : str
            file name
        **kwargs : dict
            Other parameters to make input files. See :meth:`dpdata.orca.inp.make_orca_input`
        """
        text = dpdata.orca.inp.make_orca_input(data, **kwargs)
        with open_file(file_name, "w") as fp:
            fp.write(text)
