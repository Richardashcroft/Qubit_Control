# -*- coding: utf-8 -*-
"""


#from waveform import AWG_instr, waveform_channel, IEEE_block_format
from waveform import major_channel, make_iterable, waveform
import numpy as np
from AWG520 import AWG520
from time import time, sleep
from instr import KeyboardInterruptProtection

class AWG7000(AWG520):
    """
    addr: For TCPIP, specify host addr (e.g. '192.168.1.2') with interface = 'LAN' and 
          start the VXI-11 server in the AWG 7000 instrument.
    dac_resolutions: 8 or 10. Use (8|10, 8|10) if it is channel dependent.
    """  
    
    def _selfinit(self, **kwargs):
        super(AWG7000, self)._selfinit(**kwargs)
        if 'dac_resolutions' in kwargs:
            self.dac_resolutions = kwargs['dac_resolutions']
        self.safe_write(':AWGC:RMOD SEQ')
                
    @property
    def t_sample(self):
        return self._t_sample

    @t_sample.setter
    def t_sample(self, newval):
        self._t_sample = np.clip(newval, 0.08, 100)
        self.instr.write(":FREQ %13.10f MHz" % (1000./self._t_sample))
    
    @property
    def amplitudes(self):
        return [float(self.instr.query('SOUR%d:VOLT?'%i)) for i in [1,2]]

    @amplitudes.setter
    def amplitudes(self, amps):
        amps = make_iterable(amps, repeat_len = 2)
        for i in [1,2]:
            if not 0.5 <= amps[i-1] <= 1.0:
                raise Exception('Amplitude out of range.')
            self.instr.write('SOUR%d:VOLT %f'%(i, amps[i-1]))
        
    @property
    def dac_resolutions(self):
        return [float(self.instr.query('SOUR%d:DAC:RES?'%i)) for i in [1,2]]

    @dac_resolutions.setter
    def dac_resolutions(self, resolutions):
        resol = make_iterable(resolutions, repeat_len = 2)
        for i in [1,2]:
            if not resol[i-1] in (8, 10):
                raise Exception('Invalid number of bits.')
            self.instr.write('SOUR%d:DAC:RES %f'%(i, resol[i-1]))

    def _check_wfm_length(self, wfm_length, index):
        if not wfm_length > 250:
            raise Exception("The length of waveform for AWG7000 must be longer than 250.")
        if (wfm_length%64 != 0 or wfm_length < 960) and not self.software_sequencer:
            print 'The waveform at the index of %d has %d points.'%(index+1, wfm_length)
            raise Exception("The block size for hardware sequencing in AWG7000 is 64. Allow software sequencer if you wish to.")
            
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
        
        file_list, len_list, name_list = bundle.format_TekWFM(format = 'integer')
        
        for index, wfm_length in enumerate(len_list):
            self._check_wfm_length(wfm_length, index)

        part = kwargs.get('index', slice(None))
        if part == slice(None):
            if ('id_list' in kwargs and ch_id == kwargs['id_list'][0]) or kwargs.get('clear_buffer', False):
                self.safe_write('WLISt:WAVeform:DELete ALL')
        else:
            for wfm_file, wfm_length, name in zip(file_list[part], len_list[part], name_list[part]):
                self.safe_write('WLISt:WAVeform:DELete "%s"'%(name + suffix))

        self._OPC() #complete all pending operations
        for wfm_file, wfm_length, name in zip(file_list[part], len_list[part], name_list[part]):
            info = ({'name': name + suffix, 'start_index':0, 'size': wfm_length, 'block_data': wfm_file})
#            self.safe_write('WLISt:WAVeform:NEW "{name}",{size},REAL'.format(**info))
            self.safe_write('WLISt:WAVeform:NEW "{name}",{size},INTeger'.format(**info))
            self._OPC()
            self.instr.write_raw('WLISt:WAVeform:DATA "{name}",{start_index},{size},{block_data}'.format(**info))
    
    def load_seq(self, ch_id = 'ch1', **kwargs):
        id_list = kwargs.get('id_list', [ch_id,])
        if ch_id == id_list[0]:
            self.output = False
            self.stop()
        self.send_wfms(ch_id, **kwargs)
        if ch_id == id_list[-1]:
            self._OPC()#complete all pending operations
            main_ch =  major_channel([self.ch(ch) for ch in (self.ch1_ids() + self.ch2_ids())])
            if not kwargs.get('quiet', False):
                print 'Loading the sequence into AWG ({addr})'.format(addr = self._addr),
            start = time()
            for cmd in main_ch.seq.format_TekSCPI():
               self.safe_write(cmd)
               self._OPC()#complete all pending operations
            if not kwargs.get('quiet', False):
                print 'completed in {time:.2f} seconds'.format(time = time() - start)
            self.num_of_lines = len(main_ch.seq.data)
            self.run()
    
    def run(self):
        self.safe_write(':AWGC:RUN:IMM')
        for n in [1,2]:
            self.safe_write(':OUTP%d:STAT ON'%(n))
        t0 = time()
        while self.safe_query(':AWGC:RST?') not in (u'1', u'2'):
            if time() > t0 + self.timeout:#wait up to 60 seconds
                raise Exception('AWG7000 (GPIB {addr}) is not running.'.format(addr = self._addr))
        
    def jump(self, to, ch_id = 'ch1', **kwargs):
        id_list = kwargs.get('id_list', [ch_id,])
        if ch_id == id_list[-1]:
            timeout, self.instr.timeout = self.instr.timeout, 30000 #30 sec
            t0 = time()
            while self.safe_query(':AWGC:RST?') != u'2':
                if time() > t0 + 10:#wait up to 10 seconds
                    self.instr.timeout = timeout
                    raise Exception('Cannot jump while the AWG is waiting for trigger or stopped.')
            self.instr.timeout = timeout
            self.safe_write('SEQuence:JUMP:IMMediate %d'%(to%self.num_of_lines+1))
            self._OPC()