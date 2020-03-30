# -*- coding: utf-8 -*-
"""

"""

from waveform import AWG_instr, waveform_channel, waveform
from waveform import major_channel, make_iterable, IEEE_block_format
import numpy as np
from instr import KeyboardInterruptProtection
from time import time, sleep
from visa import VisaIOError

class AWG520(AWG_instr):    
    default_seq = 'seq0'
    
    def _selfinit(self, **kwargs):
        self.instr.read_termination = '\n'
        self.instr.timeout = 30000 #10 sec
        self.timeout = 60000 #timeout for loading etc.

        self.t_sample  = kwargs.get('t_sample', 0.1)
        self.ref_clock = kwargs.get('ref_clock', 'Int')

        self.catalog_seq, self.catalog_wfm = {}, {}
        self.ch1 = waveform_channel(instr = self, ch_id = 'ch1')
        self.ch2 = waveform_channel(instr = self, ch_id = 'ch2')
        self.mk11 = waveform_channel(instr = self, ch_id = 'mk11')
        self.mk12 = waveform_channel(instr = self, ch_id = 'mk12')
        self.mk21 = waveform_channel(instr = self, ch_id = 'mk21')
        self.mk22 = waveform_channel(instr = self, ch_id = 'mk22')
        
        self.scales = kwargs.get('scales', 1.)
        self.amplitudes = kwargs.get('amplitudes', 1.)
        
        self.software_sequencer = kwargs.get('software_sequencer', False)
        self.safe_write(':AWGC:RMOD ENH')
        self.num_of_lines = 0
        
        self.errors = [self.safe_query(':SYST:ERR?')]
        while not self.errors[-1].startswith('0'):
            self.errors.append(self.safe_query(':SYST:ERR?'))
        if len(self.errors) > 1:
            print 'AWG ({addr}) has {num} error messages.'.format(addr = self._addr, num = len(self.errors) -1)
            if len(self.errors) > 21:
            	print 'Printing most recent errors. All error messages are stored in self.errors.'
        	print self.errors[-21:-1]
    
    @property
    def t_sample(self):
        return self._t_sample

    @t_sample.setter
    def t_sample(self, newval):
        self._t_sample = np.clip(newval, 1., 2e+4)
        self.safe_write(":FREQ %13.10f MHz" % (1000./self._t_sample))
    
    @property
    def ref_clock(self):
    	return self.safe_query(':ROSCillator:SOURce?')

    @ref_clock.setter
    def ref_clock(self, source):
    	self.safe_write(':ROSCillator:SOURce %s' % source)

    @property
    def amplitudes(self):
        return [float(self.safe_query('SOUR%d:VOLT?'%i)) for i in [1,2]]

    @amplitudes.setter
    def amplitudes(self, amps):
        amps = make_iterable(amps, repeat_len = 2)
        for i in [1,2]:
            if not 0.1 <= amps[i-1] <= 2.0:
                raise Exception('Amplitude out of range.')
            self.safe_write('SOUR%d:VOLT %f'%(i, amps[i-1]))
    
    @property
    def scales(self):
        return [self.ch1.scale, self.ch2.scale]
    
    @scales.setter
    def scales(self, scales):
        scales = make_iterable(scales, repeat_len = 2)
        self.ch1.scale, self.ch2.scale = scales[0], scales[1]

    @property
    def output(self):
        return self._output
    
    @output.setter
    def output(self, on_off):
        for n in [1,2]:
            self.safe_write(':OUTP%d:STAT OFF'%(n))
        self._output = False

    

    def ch1_ids(self, id_list = ['ch1', 'mk11', 'mk12']):
        return [_id for _id in id_list if _id in ['ch1', 'mk11', 'mk12']]
    
    def ch2_ids(self, id_list = ['ch2', 'mk21', 'mk22']):
        return [_id for _id in id_list if _id in ['ch2', 'mk21', 'mk22']]    
    
    def ch(self, ch_id):
        if ch_id in self.ch1_ids() or ch_id in self.ch2_ids():
            return getattr(self, ch_id)
    
    def send_wfms(self, ch_id, **kwargs):
        id_list = kwargs.get('id_list', [ch_id,])
        if ch_id in self.ch1_ids() and ch_id == self.ch1_ids(id_list)[0]:
            bundle = waveform([self.ch1, self.mk11, self.mk12])
            suffix = '_ch1'
        elif ch_id in self.ch2_ids() and ch_id == self.ch2_ids(id_list)[0]:
            bundle = waveform([self.ch2, self.mk21, self.mk22])
            suffix = '_ch2'
        else:
            return
        
        file_list, len_list, name_list = bundle.format_MAGIC1000()
        
        for index, wfm_length in enumerate(len_list):
            self._check_wfm_length(wfm_length, index)
        
        for name, wfm_file in zip(name_list, file_list):
            self._OPC()
            self.instr.write_raw(':MMEM:DATA "%s%s.wfm",%s' % (name, suffix, IEEE_block_format(wfm_file)))

    def _check_wfm_length(self, wfm_length, index):
        if not (wfm_length >= 400 and wfm_length%4 == 0):
            print 'The waveform at the index of %d has %d points.'%(index+1, wfm_length)
            raise Exception("The length of waveform for AWG520 must be longer than 400 and a multiple of 4.")
    
    def send_seq(self, **kwargs):
        main_ch =  major_channel([self.ch(ch_id) for ch_id in (self.ch1_ids() + self.ch2_ids())])
        seq_file= main_ch.seq.format_MAGIC3002()
        seq_file_name = '%s.seq'%kwargs.get('seq_name', self.default_seq)
        
        with KeyboardInterruptProtection():
             self.instr.write_raw(':MMEM:DATA "%s",%s'%(seq_file_name, IEEE_block_format(seq_file)))
        
        if kwargs.get('load_after_send', False):
            if not kwargs.get('quiet', False):
                print 'Loading the sequence into AWG ({addr})'.format(addr = self._addr)
            self._OPC()#complete all pending operations
            self.safe_write(':FUNC:USER "%s"'%seq_file_name)
            self._OPC()#complete all pending operations
            self.num_of_lines = len(main_ch.seq.data)
    
    def load_seq(self, ch_id = 'ch1', **kwargs):
        id_list = kwargs.get('id_list', [ch_id,])
        if ch_id == id_list[0]:
            self.stop()
        self.send_wfms(ch_id, **kwargs)
        if ch_id == id_list[-1]:
            self.send_seq(load_after_send = True, **kwargs)
            self.run()

    def run(self):
        self.safe_write(':AWGC:RUN:IMM')
        t0 = time()
        while int(self.safe_query(':AWGC:RST?')) == 0:
            if time() > t0 + 120:
                raise Exception('AWG520 (GPIB {addr}) is not running.'.format(addr = self._addr))
        for n in [1,2]:
            self.safe_write(':OUTP%d:STAT ON'%(n))
        while not int(self.safe_query(':OUTP%d:STAT?'%(n))) == 1:
            if time() > t0 + 120:
                raise Exception('AWG520 (GPIB {addr}) does not output.'.format(addr = self._addr))            
    
    def stop(self):
         self.safe_write(':AWGC:STOP')
    
    def reset(self):
        self.safe_write('*RST')

    def restart(self):
        self.safe_write('AWGControl:STOP')
        self.run()

    def _OPC(self):
        start = time()
        while time() - start < self.timeout:
            try:
                if self.safe_query('*OPC?') == u'1':
                    return
            except VisaIOError:
                pass
            sleep(0.01)
        raise Exception('Cannot complete pending operations.')

#    def get_seq(self, **kwargs):
#        seq_file_name = '%s.seq'%kwargs.get('seq_name', self.default_seq)
#        self.safe_write('MMEMory:DATA?%s'%seq_file_name)
#        content = [self.safe_read(),]
#        while len(content[-1]):
#            content.append(self.safe_read())
    
    def jump(self, to, ch_id = 'ch1', **kwargs):
        id_list = kwargs.get('id_list', [ch_id,])        
        if ch_id == id_list[-1]:
            #self.safe_write('AWGC:ENH:SEQ:JMOD SOFT')
            self.safe_write('AWGC:EVEN:SOFT:IMM %d'%(to%self.num_of_lines+1))