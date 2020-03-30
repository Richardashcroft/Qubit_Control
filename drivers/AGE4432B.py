# -*- coding: utf-8 -*-
"""

"""
from instr import container, variable


class AGE4432B(container):

    def _selfinit(self, **kwargs):
        self.safe_write("OUTP:STAT ON")
        self.safe_write("UNIT:POW DBM")

    @variable(type=float, min = -136, max = 13, units='dBm')
    def power(self):
        return self.safe_query("POW:LEV?")

    @power.setter
    def power(self, setval):
        self.safe_write("POW:LEV %f DBM"%setval)

    @variable(min = 0.25, max = 3000., units='MHz')
    def frequency(self):
        return float(self.safe_query("FREQ:CW?"))/1e+6

    @frequency.setter
    def frequency(self, setval):
        self.safe_write("FREQ:CW %12.9f MHZ" % setval)
