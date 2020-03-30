# -*- coding: utf-8 -*-
"""

"""
from instr import container, meas, var, measurable, variable
import numpy as np

class Keithley2450(container):
    """ A container to configure a Keithley2450 in V-source & I-measure
   
        name: display name for the variable
        step (0.01): maximum allowed step size when the variable value is moved
        min (-np.inf): minimum value of the variable
        max (np.inf): maximum value of the variable
        wait (0): wait time (sec) when the variable value is moved
        average (1): number of averaging (repeat)
        nplc (1): integration time in number of power line cylces
        #It is strongly recommended to use an integer nplc to reduce the noise.
    """
    def _selfinit(self, **kwargs):
        self.step = kwargs.get('step', 0.01)
        self.voltage = Keithley2450_voltage_source(owner = self, **kwargs)
        self.current = Keithley2450_current_measure(owner = self, **kwargs)

class Keithley2450_source_I_measure_V(container):
    """  A container to configure a Keithley2450 in I-source & V-measure
   
        name: display name for the variable
        step (1uA): maximum allowed step size when the variable value is moved
        min (-np.inf): minimum value of the variable
        max (np.inf): maximum value of the variable
        wait (0): wait time (sec) when the variable value is moved
        average (1): number of averaging (repeat)
        nplc (1): integration time in number of power line cylces
        #It is strongly recommended to use an integer nplc to reduce the noise.
    """
    def _selfinit(self, **kwargs):
        self.step = kwargs.get('step', 1e-6)
        self.current = Keithley2450_current_source(owner = self, **kwargs)
        self.voltage = Keithley2450_voltage_measure(owner = self, **kwargs)

class Keithley2450_voltage_source(var):
    """
    """ 
    def _selfinit(self, **kwargs):
        if not self.instr.query("*LANG?").startswith("SCPI"):
            raise Exception("Set the command set to SCPI and reboot the instrument.")

        vranges = [0.02, 0.2, 2, 20, 200, np.nan]
        self.range = [vr for vr in vranges if vr >= kwargs.get("range", np.nan) or np.isnan(vr)][0]

        self.instr.write("SOUR:FUNC VOLT")
        if np.isnan(self.range):
            self.instr.write("SOUR:VOLT:RANG:AUTO ON")
        else:
            self.instr.write("SOUR:VOLT:RANG %f" % self.range)
        if "compliance" in kwargs:
            self.isntr.write("SOUR:VOLT:ILIM %f" % kwargs["compliance"])
        self.instr.write("OUTP ON")

    def _set(self, destval):
        self.instr.write("SOUR:VOLT %.8f" % destval)      
      
    def _get(self):
        return self.instr.query_ascii_values("SOUR:VOLT?")[0]

class Keithley2450_voltage_measure(meas):
    """
    """    
    @property
    def nplc(self):
        return self._nplc

    @nplc.setter
    def nplc(self, newval):
        self._nplc = np.clip(newval, 0.01, 10)
        self.instr.write("SENS:VOLT:NPLC %f"% self.nplc)
    
    @property
    def avg(self):
        return self._avg

    @avg.setter
    def avg(self, newval):
        self._avg = np.clip(newval, 1, 100)
        self.instr.write("SENS:VOLT:AVER:TCON REP") # MOVing filter is also available.
        self.instr.write("SENS:VOLT:AVER:STAT ON")
        self.instr.write("SENS:VOLT:AVER:COUN %d" % self.avg)
    
    def _selfinit(self, **kwargs):
        if not self.instr.query("*LANG?").startswith("SCPI"):
            raise Exception("Set the command set to SCPI and reboot the instrument.")

        vranges = [0.02, 0.2, 2, 20, 200, np.nan]
        self.range = [vr for vr in vranges if vr >= kwargs.get("range", np.nan) or np.isnan(vr)][0]

        self.avg = kwargs.get("average", 1)
        self.nplc = kwargs.get("nplc", 1)
        self.instr.write('SENS:FUNC "VOLT"')
        self.instr.write('TRAC:CLE')
        self.instr.write(":TRAC:FILL:MODE CONT")
        
        if np.isnan(self.range):
             self.instr.write("SENS:VOLT:RANG:AUTO ON")
        else:
             self.instr.write("SENS:VOLT:RANG %f" % self.range)

    def _get(self):
        return self.instr.query_ascii_values("MEAS:VOLT?")[0]
        
class Keithley2450_current_source(var):
    def _selfinit(self, **kwargs):
        if not self.instr.query("*LANG?").startswith("SCPI"):
            raise Exception("Set the command set to SCPI and reboot the instrument.")

        iranges = [10e-9, 100e-9, 1e-6, 10e-6, 100e-6, 1e-3, 10e-3, 100e-3, 1, np.nan]
        self.range = [ir for ir in iranges if ir >= kwargs.get("range", np.nan) or np.isnan(ir)][0]

        self.instr.write("SOUR:FUNC CURR")
        if np.isnan(self.range):
            self.instr.write("SOUR:CURR:RANG:AUTO ON")
        else:
            self.instr.write("SOUR:CURR:RANG %f" % self.range)
        if "compliance" in kwargs:
            self.isntr.write("SOUR:CURR:VLIM %f" % kwargs["compliance"])
        self.instr.write("OUTP ON")

    def _set(self, setval):
        self.instr.write("SOUR:CURR %.14f" % setval)

    def _get(self):
        return self.instr.query_ascii_values("SOUR:CURR?")[0]

class Keithley2450_current_measure(meas):
    """
    """    
    @property
    def nplc(self):
        return self._nplc

    @nplc.setter
    def nplc(self, newval):
        self._nplc = np.clip(newval, 0.01, 10)
        self.instr.write("SENS:CURR:NPLC %f"% self.nplc)
    
    @property
    def avg(self):
        return self._avg

    @avg.setter
    def avg(self, newval):
        self._avg = np.clip(newval, 1, 100)
        self.instr.write("SENS:CURR:AVER:TCON REP") # MOVing filter is also available.
        self.instr.write("SENS:CURR:AVER:STAT ON")
        self.instr.write("SENS:CURR:AVER:COUN %d" % self.avg)
    
    def _selfinit(self, **kwargs):
        if not self.instr.query("*LANG?").startswith("SCPI"):
            raise Exception("Set the command set to SCPI and reboot the instrument.")

        iranges = [10e-9, 100e-9, 1e-6, 10e-6, 100e-6, 1e-3, 10e-3, 100e-3, 1, np.nan]
        self.range = [ir for ir in iranges if ir >= kwargs.get("range", np.nan) or np.isnan(ir)][0]

        self.avg = kwargs.get("average", 1)
        self.nplc = kwargs.get("nplc", 1)
        self.instr.write('SENS:FUNC "CURR"')
        self.instr.write('TRAC:CLE')
        self.instr.write(":TRAC:FILL:MODE CONT")

        if np.isnan(self.range):
            self.instr.write("SENS:VOLT:RANG:AUTO ON")
        else:
            self.instr.write("SENS:VOLT:RANG %f" % self.range)

    def _get(self):
        ret = self.instr.query_ascii_values("MEAS:CURR?")[0]
        return ret
