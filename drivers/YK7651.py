# -*- coding: utf-8 -*-
"""

"""

from instr import container, meas, var, measurable, variable

class YK7651(container):
    """
    range ('10V'): '10mV', '100mV', '1V', '10V', '30V'
    """
    def _selfinit(self, **kwargs):
        self.safe_write('O1E')
        self._rng_dict = {'10mV':2,  '100mV':3, '1V':4, '10V':5, '30V':6}
        self.range = kwargs.get('range', '10V')
    
    @property
    def range(self):
        return self._range
    
    @range.setter
    def range(self, rng_str):
        if not rng_str in self._rng_dict:
            raise Exception('Choose the range from %s.'%(', '.join(self._rng_dict.keys())))
        self.safe_write('R%dE')
        self._range = rng_str
    
    @variable(type = float, units = 'Volt')
    def voltage(self):
        self._voltage = float(self.safe_query('OD')[4:-1])
        return self._voltage
    
    @voltage.setter
    def voltage(self, volt):
        rng_max_dict = {'10mV': 0.12, '100mV': 0.12, '1V':1.2, '10V':12., '30V':32.}
        if volt > rng_max_dict[self.range]:
            raise Exception('Specified output (%f) exceeds the maximum level (%f).'%(volt, rng_max_dict[self.range]))
        
        self.safe_write('F1R%dS%fE'%(self._rng_dict[self.range], volt))
        self._voltage = volt