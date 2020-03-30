# -*- coding: utf-8 -*-
"""



from instr import container, meas, var, measurable, variable
from numpy import asarray, linspace, logspace, log10
from struct import pack

class AG4407B(container):
    def _selfinit(self, **kwargs):
        #self.safe_write(':FREQuency:CENTer:STEP:AUTO ON') #frequency center stepping auto
        self.attenuation = kwargs.get('attenuation', 10)
        self.bandwidth = kwargs.get('bandwidth', self.bandwidth)
        self.start = kwargs.get('start', self.start)
        self.stop = kwargs.get('stop', self.stop)
        self.points = kwargs.get('points', self.points)
        if 'frequency' in kwargs:
            self.frequency = kwargs['frequency']
        self.safe_write(':FORMat ASCii')

    def clear_trace(self):
        nan_value = -777
        self.safe_write('TRACe TRACE1, %s'%','.join([str(nan_value)]*int(self.points)))
        self.safe_write(':INITiate')

    @property
    def attenuation(self):
        return float(self.safe_query(':POWer:ATTenuation?'))
    
    @attenuation.setter
    def attenuation(self, attn):
        self.safe_write(':POWer:ATTenuation:AUTO OFF') #input attenuator auto OFF
        self.safe_write(':POWer:ATTenuation %f'%attn)
    
    @property
    def bandwidth(self):#resolution bandwidth
        return float(self.safe_query(':BANDwidth?'))
    
    @bandwidth.setter
    def bandwidth(self, bw):
        self.safe_write(':BANDwidth %15.13e'%bw)
        self.safe_write(':BANDwidth:VIDeo:AUTO')
    
    @property
    def start(self):
        self._start = float(self.safe_query(':FREQuency:STARt?'))
        return self._start
    
    @start.setter
    def start(self, start):
        if not -80e+6 <= start <= 27e+9:
            raise Exception('Frequency out of range')
        if start == self._stop if hasattr(self, '_stop') else self.stop:
            self.safe_write(':FREQuency:CENTer %15.14f' %start)
            self.safe_write(':FREQuency:SPAN 0')
        else:
            self.safe_write(':FREQuency:STARt %15.14f' %start)
        self._start = self.start
    
    @property
    def stop(self):
        self._stop = float(self.safe_query(':FREQuency:STOP?'))
        return self._stop
    
    @stop.setter
    def stop(self, stop):
        if not -80e+6 <= stop <= 27e+9:
            raise Exception('Frequency out of range')
        if stop == self._start if hasattr(self, '_start') else self.start:
            self.safe_write(':FREQuency:CENTer %15.14f' %stop)
            self.safe_write(':FREQuency:SPAN 0')
        else:
            self.safe_write(':FREQuency:STOP %15.14f' %stop)
        self._stop = self.stop
    
    @property
    def points(self):
        self._points = float(self.safe_query(':SWEep:POINts?'))
        return self._points
    
    @points.setter
    def points(self, points):
        minpnts = 2 if self._start == self._stop else 101
        if not minpnts <= points <= 8192:
            raise Exception('Points out of range')
        self.safe_write(':SWEep:POINts %d' %points)
        self._points = self.points
    
    @property
    def sweep_spacing(self):
        self._sweep_spacing = self.safe_query(':SWEep:SPACing?')[:3].lower()
        return self._sweep_spacing
    
    @sweep_spacing.setter
    def sweep_spacing(self, spacing):
        if spacing[:3].lower() in ('lin', 'log'):
            self.safe_write(':SWEep:SPACing %s'%spacing[:3].upper())
        else:
            raise Exception('No sweep spacing matched.')
        self._sweep_spacing = self.sweep_spacing
    
    @property
    def frequency(self):
        start, stop = self.start, self.stop
        if start == stop:
            return start
        elif self.sweep_spacing == 'lin':
            return linspace(start, stop, self.points)
        else:
            return logspace(log10(start), log10(stop), self.points)
    
    @frequency.setter
    def frequency(self, freq):
        if hasattr(freq, '__iter__'):
            start, stop, points = freq[0], freq[-1], len(freq)
            if all(freq == linspace(start, stop, points)) and start != stop:
                self.sweep_spacing = 'lin'
            elif all(freq == logspace(log10(start), log10(stop), points)) and start != stop:
                self.sweep_spacing = 'log'
            else:
                raise Exception('No sweep spacing matched.')
            self.start, self.stop, self.points = start, stop, points
        else:
            self.start, self.stop = freq, freq
        self._frequency = self.frequency
        
    @property
    def trace(self):
        return self.acquire_trace()
    
    def acquire_trace(self, clear_buffer = True):
        if clear_buffer:
            self.clear_trace()
        self.safe_write('TRACe? TRACE1')
        trace =  asarray([float(s) for s in self.safe_read().split(',')])
        if len(trace) != self.points or any(trace < -500.):
            return self.acquire_trace(clear_buffer = False)
        else:
            return trace