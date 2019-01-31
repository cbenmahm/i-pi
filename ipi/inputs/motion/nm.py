"""Deals with creating the ensembles class.

Copyright (C) 2013, Joshua More and Michele Ceriotti

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <http.//www.gnu.org/licenses/>.

Inputs created by Michele Ceriotti and Benjamin Helfrecht, 2015

Classes:
   InputGeop: Deals with creating the Geop object from a file, and
      writing the checkpoints.
"""

import numpy as np
import ipi.engine.initializer
from ipi.engine.motion import *
from ipi.utils.inputvalue import *
from ipi.inputs.thermostats import *
from ipi.inputs.initializer import *
from ipi.utils.units import *

__all__ = ["InputNormalMode"]


class InputNormalMode(InputDictionary):
    """Dynamic matrix calculation options.

       Contains options related with finite difference computation of force constats. 

    """

    attribs = {"mode": (InputAttribute, {"dtype": str, "default": "imf",
                                         "help": "The algorithm to be used: independent mode framework (imf) and vibrational self consistent field (vscf).",
                                         "options": ["imf", "vscfmapper", "vscfsolver", "pc"]})}
    fields = {
                "prefix": (InputValue, {"dtype": str, "default": "PHONONS",
                                        "help": "Prefix of the output files."
                                        }),
                "asr": (InputValue, {"dtype": str, "default": "none", "options": ["none", "poly", "lin", "crystal"],
                                     "help": "Removes the zero frequency vibrational modes depending on the symmerty of the system."
                                     }),
                "dynmat": (InputArray, {"dtype": float,
                                        "default": np.zeros(0, float),
                                        "help": "Portion of the dynamical matrix known up to now."}),
                "nprim": (InputValue, {"dtype": float,
                                        "default": 1.0,
                                        "help": "Number of primitive unit cells in the simulation cell."}),
                "fnmrms": (InputValue, {"dtype": float,
                                        "default": 1.0,
                                        "help": "Fraction of harm RMS displ used to sample along normal mode."}),
                "nevib": (InputValue, {"dtype": float,
                                        "default": 25.0,
                                        "help": "Multiple of harm vibr energy up to which BO surface is sampled."}),
                "nint": (InputValue, {"dtype": int,
                                        "default": 101,
                                        "help": "Integration points for Hamiltonian matrix elements."}),
                "nbasis": (InputValue, {"dtype": int,
                                        "default": 10,
                                        "help": "Number of SHO states used as basis for anharmonic wvfn."}),
                "athresh": (InputValue, {"dtype": float,
                                        "default": 1e-2,
                                        "help": "Convergence thresh for fractional error in vibr free energy."}),
                "ethresh": (InputValue, {"dtype": float,
                                        "default": 1e-2,
                                        "help": "Convergence thresh for fractional error in vibr free energy."}),
                "nkbt": (InputValue, {"dtype": float,
                                        "default": 4.0,
                                        "help": "Threshold for (e - e_gs)/(kB T) of vibr state to be incl in the VSCF and partition function."}),
                "nexc": (InputValue, {"dtype": int,
                                        "default": 5,
                                        "help": "Minimum number of excited n-body states to calculate (also in MP2 correction)."}),
                "mptwo": (InputValue, {"dtype": bool,
                                        "default": False,
                                        "help": "Flag determining whether MP2 correction is calculated."}),
                "print_mftpot": (InputValue, {"dtype": bool,
                                        "default": False,
                                        "help": "Flag determining whether MFT potentials are printed to file."}),
                "print_2b_map": (InputValue, {"dtype": bool,
                                        "default": False,
                                        "help": "Flag determining whether mapped coupling potentials are printed to file."}),
                "print_vib_density": (InputValue, {"dtype": bool,
                                        "default": False,
                                        "help": "Flag determining whether the vibrational density (|psi|^2) are printed to file."}),
                "threebody": (InputValue, {"dtype": bool,
                                        "default": False,
                                        "help": "Flag determining whether three-mode coupling terms are accounted for."}),
                "refdynmat": (InputArray, {"dtype": float,
                                           "default": np.zeros(0, float),
                                           "help": "Portion of the refined dynamical matrix known up to now."})
    }

    dynamic = {}

    default_help = "Fill in."
    default_label = "PHONONS"

    def store(self, nm):
        if nm == {}: return
        self.mode.store(nm.mode)
        self.prefix.store(nm.prefix)
        self.asr.store(nm.asr)
        self.dynmat.store(nm.dynmatrix)
        self.nprim.store(nm.nprim)
        self.refdynmat.store(nm.refdynmatrix)
        self.fnmrms.store(nm.fnmrms) #"1.0"
        self.nevib.store(nm.nevib) #"25.0"
        self.nint.store(nm.nint) #"101"
        self.nbasis.store(nm.nbasis) #"10"
        self.athresh.store(nm.athresh) #"1e-2"
        self.ethresh.store(nm.ethresh) #"1e-2"
        self.nkbt.store(nm.nkbt) #"4.0"
        self.nexc.store(nm.nexc) #"5"
        self.mptwo.store(nm.mptwo) #False
        self.print_mftpot.store(nm.print_mftpot) #False
        self.print_2b_map.store(nm.print_2b_map) #False
        self.print_vib_density.store(nm.print_vib_density) #False
        self.threebody.store(nm.threebody) #False

    def fetch(self):
        rv = super(InputNormalMode, self).fetch()
        rv["mode"] = self.mode.fetch()
        return rv