# -*- coding: utf-8 -*-
"""

"""
from instr import container, variable

"""
This is identical to AG4432B.py.
"""
class AG83650B(container):

    def _selfinit(self, **kwargs):
        self.safe_write("OUTP:STAT ON")
        self.safe_write("UNIT:POW DBM")

    @variable(type=float, min = -110, max = 10, units='dBm')
    def power(self):
        return self.safe_query("POW:LEV?")

    @power.setter
    def power(self, setval):
        self.safe_write("POW:LEV %f DBM"%setval)

    @variable(min = 10, max = 50000., units='MHz')
    def frequency(self):
        return float(self.safe_query("FREQ:CW?"))/1e+6

    @frequency.setter
    def frequency(self, setval):
        self.safe_write("FREQ:CW %12.9f MHZ" % setval)
