
from instr import container, meas, var, measurable, variable

class AG3631A(container):
    def _selfinit(self, **kwargs):
        self.ch1 = AG3631A_ch(owner = self, ch = 1, **kwargs) #6V output
        self.ch2 = AG3631A_ch(owner = self, ch = 2, **kwargs) #+25V output
        self.ch3 = AG3631A_ch(owner = self, ch = 3, **kwargs) #-25V output
        self.safe_write('OUTP:STAT ON')

class AG3631A_ch(container):
    """
    ch: 1 ('6V'), 2 ('+25V'), 3 ('-25V')
    compliance (max)
    """
    def _selfinit(self, **kwargs):
        self.ch_num = kwargs['ch']
        self.ch_name = ['P6V', 'P25V', 'N25V'][self.ch_num-1]
        condition = int(self.safe_query('STAT:QUES:INST:ISUM%d:COND?'%self.ch_num))
        if condition in [1,2]:
            self.mode = 'CV' if condition is 2 else 'CC'
        else:
            raise Exception('The instrument is either off or in the hardware failure.')
        max = [6., 25., -25.] [self.ch_num - 1] if self.mode == 'CV' else [5., 1., 1.] [self.ch_num - 1]
        self.compliance = kwargs.get('compliance', max)
    
    @variable(type = float, units = 'V')
    def voltage(self):
        self._voltage = float(self.query('MEAS? %s'%self.ch_name))
        return self._voltage
        
    @voltage.setter
    def voltage(self, volt):
        if self.mode is 'CV':
            self.safe_write('APPL %s, %f, %f'%(self.ch_name, volt, self.compliance))
            self._voltage = volt
        else:
            raise Exception('The instrument is in the constant current mode.')
    
    @variable(type = float, units = 'A')
    def current(self):
        self._current = float(self.query('MEAS:CURR? %s'%self.ch_name))
        return self._current
    
    @current.setter
    def current(self, curr):
        if self.mode is 'CC':
            self.safe_write('APPL %s, %f, %f'%(self.ch_name, self.compliance, curr))
            self._current = curr
        else:
            raise Exception('The instrument is in the constant voltage mode.')
