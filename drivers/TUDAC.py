# -*- coding: utf-8 -*-
"""

"""
from instr import container, var, KeyboardInterruptProtection
from pyvisa import constants, VisaIOWarning

import warnings
# Ignore VisaIOWarning warning in pyvisa module during the serial communication
warnings.filterwarnings('ignore', 'VI_SUCCESS_MAX_CNT.*', VisaIOWarning, 'pyvisa')


class TUDAC(container):
    """
    Channels (16): Number of total channels in the TUDAC, either 8 or 16.
    Polarity (Bipolar): Common to all channels Negative/Positive/Bipolar.
    """

    def _selfinit(self, **kwargs):
        self.instr.timeout = 1000
        self.num_channels = kwargs.get('Channels', 16)
        polarity = {'Negative':(-4.0,0.0), 'Positive':(0.0,4.0), 'Bipolar':(-2.0,2.0)}
        self.lower_lim, self.upper_lim = polarity[kwargs.get('Polarity', 'Bipolar')]

        #Configure Serial communication:
        self.instr.baud_rate = 115200
        self.instr.data_bits = 8
        self.instr.stop_bits = constants.StopBits.one
        self.instr.parity = constants.Parity.odd
        self.instr.end_input = constants.SerialTermination.none

        for ch_num in range(self.num_channels):
            c = TUDAC_ch(ch=ch_num+1, units='V', owner=self, **kwargs)
            setattr(self, "ch%d"%(ch_num+1), c)

class TUDAC_ch(var):

    def _selfinit(self, **kwargs):
        self._ch = kwargs['ch']

    @property
    def min(self):
        return self._min

    @min.setter
    def min(self, newval):
        self._min = max(newval, self.lower_lim)

    @property
    def max(self):
        return self._max

    @max.setter
    def max(self, newval):
        self._max = min(newval, 2.)

    def _get(self):
        with KeyboardInterruptProtection():
            cmd = "%c%c%c%c" % (4, 0, self.num_channels*2+2, 2)
            self.lib.write(self.instr.session, cmd)

            data1, _ = self.lib.read(self.instr.session, 2)
            read_len = ord(data1[0])

            data2, _ = self.lib.read(self.instr.session, read_len - 2)
        return self.decode_volt([ord(s) for s in data2])

    def _set(self, setvalue):
        with KeyboardInterruptProtection():
            data_H, data_L = self.encode_volt(setvalue)
            cmd = "%c%c%c%c%c%c%c" % (7, 0, 2, 1, self._ch, data_H, data_L)
            self.lib.write(self.instr.session, cmd)

            data1, _ = self.lib.read(self.instr.session, 2)
            if ord(data1[1]) not in (0, 32):
                print "Error(TUDAC): Failed to read echo after setting voltage: %s" % data1

            data2, _ = self.lib.read(self.instr.session, ord(data1[0]) - 2) #echo -- just discarding the response

    @property
    def status(self):
        if self.encode_volt(self._value) == self.encode_volt(float('%.4f'%self._value)):
            numparts = ("%s = %.4f"%(self.name, self._value)).rstrip('0')
        else:            
            numparts = ("%s = %.5f"%(self.name, self._value)).rstrip('0')
        return '%s %s'%(numparts, self.units)

    def decode_volt(self, code):
        try:
            idx = self._ch - 1
            return (self.upper_lim - self.lower_lim) * (code[2*idx] * 0x0100 + code[2*idx+1]) / 0xffff + self.lower_lim
        except:
            return float('nan')

    def encode_volt(self, volt):
        try:
            int_code = int(round(0xffff * (volt - self.lower_lim) / (self.upper_lim - self.lower_lim)))
            return int_code // 0x0100, int_code % 0x0100
        except:
            return None, None