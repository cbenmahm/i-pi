"""Interface with librascal to run machine learning potentials"""

import sys
from .dummy import Dummy_driver

from ipi.utils.mathtools import det_ut3x3
from ipi.utils.units import unit_to_internal, unit_to_user

try:
    from rascal.models.genericmd import FiniteTCalculator as RascalCalc
except:
    RascalCalc = None


class Finite_T_driver(Dummy_driver):
    def __init__(self, args=None):

        if RascalCalc is None:
            raise ImportError("Couldn't load librascal bindings")

        self.error_msg = """Finite_T driver requires specification of a .json (dummy) model file fitted with librascal
                            containing information about the active set (sparse points), 
                            a template file that describes the chemical makeup of the structure,
                            and the DOS regression wights. 
                            Example: 
                            python driver.py -m rascal -u -o model.json,template.xyz,xdos.npy,temperature,is_volume,nelectrons"""

        super().__init__(args)

    def check_arguments(self):
        """Check the arguments required to run the driver

        This loads the potential and atoms template in librascal, calculates the new weights required to 
        compute the finite temperature contribution to the total force and stress
        """
        try:
            arglist = self.args.split(",")
        except ValueError:
            sys.exit(self.error_msg)

        if len(arglist) == 5:
            self.model = arglist[0]
            self.template = arglist[1]
            self.xdos = arglist[2]
            self.temperature = arglist[3]
            self.nelectrons = arglist[4] 
        else:
            sys.exit(self.error_msg)

        self.base_calc = RascalCalc(self.model, True, self.xdos, self.temperature, self.template, self.nelectrons)

    def __call__(self, cell, pos):
        """Get energies, forces, and stresses from the librascal model"""
        pos_rascal = unit_to_user("length", "angstrom", pos)
        cell_rascal = unit_to_user("length", "angstrom", cell)
        # Do the actual calculation
        pot, force, stress, extras = self.base_calc.calculate(pos_rascal, cell_rascal)
        pot_ipi = unit_to_internal("energy", "electronvolt", pot)
        force_ipi = unit_to_internal("force", "ev/ang", force)
        # The rascal stress is normalized by the cell volume (in rascal units)
        vir_rascal = -1 * stress * det_ut3x3(cell_rascal)
        vir_ipi = unit_to_internal("energy", "electronvolt", vir_rascal)
        extras = ""
        return pot_ipi, force_ipi, vir_ipi, extras
