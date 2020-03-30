# -*- coding: utf-8 -*-
"""

"""
from instr import container, variable, measurable
import re


class AG4142B(container):

    def _selfinit(self, **kwargs):
        self.instr.write_termination = '\r\n'
        self.instr.read_termination = '\r\n'
        
        if "ch_num" in kwargs:
            self.ch = kwargs["ch_num"]
        else:
            raise Exception("Please specify the ch number as ch_num.")
        if "compliance" in kwargs:
            self.compliance = kwargs["compliance"]
        else:
            self.compliance = 1.0e-3 #default value = 1e-3
        if "range_val" in kwargs:
            self.range = kwargs["range_val"]
        else:
            self.range = 0 #default value = 0 (auto range)
        if "ave" in kwargs:
            self.average = kwargs["ave"]
        else:
            self.average = 128 #default value = 128
        """
        if "mode" in kwargs:
            self.mode = kwargs["mode"] #mode = 2: differential voltage measurement (only for VM)
        else:
            self.mode = 0 #default value = 0
        """
        self.safe_write("FMT1")
        self.safe_write("CN %d" %self.ch)
        
        if self.ch > 8: #for VS, VM
            self.safe_write("VM %d ,1" %self.ch)
        else:
            pass
        
    
    @variable(type=float, min = -100, max = 100, units='V')
    def voltagevar(self):
        value = re.split(",", self.safe_query("*LRN? %d" %self.ch))[2]
        return float(value)

    @voltagevar.setter
    def voltagevar(self, setval):
        self.safe_write("DV %d , %d, %f , %f" %(self.ch, self.range, setval, self.compliance))
    
    
    @variable(type=float, min = -100, max = 100, units='A') #need to search correct min and max values
    def currentvar(self):
        value = re.split(",", self.safe_query("*LRN? %d" %self.ch))[2]
        return float(value)

    @currentvar.setter
    def currentvar(self, setval):
        self.safe_write("DI %d , %d, %f , %f" %(self.ch, self.range, setval, self.compliance))
    
     
    @measurable(type=float, units='V')
    def voltagemeas(self):
        self.safe_write("MM1, %d" %self.ch)
        self.safe_write("RV %d , %d" %(self.ch, self.range))
        self.safe_write("AV %d , 0" %self.average)
        return float(self.safe_query("XE")[3:19])
    
    @measurable(type=float, units='A')
    def currentmeas(self):
        self.safe_write("MM1, %d" %self.ch)
        self.safe_write("RI %d , %d" %(self.ch, self.range))
        self.safe_write("AV %d , 0" %self.average)
        return float(self.safe_query("XE")[3:19])