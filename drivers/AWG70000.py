# -*- coding: utf-8 -*-
"""

"""

#from waveform import AWG_instr, waveform_channel, IEEE_block_format
from waveform import major_channel, make_iterable, waveform, waveform_channel
import numpy as np
from AWG520 import AWG520
from time import time, sleep
from instr import KeyboardInterruptProtection

class AWG70000(AWG520):
    """
    addr: For TCPIP, specify host addr (e.g. '192.168.1.2') with interface = 'LAN' and 
          start the VXI-11 server in the AWG 7000 instrument.
    dac_resolutions: 
    """  
    
    def _selfinit(self, **kwargs):
        self.instr.read_termination = '\n'
        self.instr.timeout = 30000 #10 sec
        self.timeout = 60000 #timeout for loading etc.

        self.t_sample = kwargs.get('t_sample', 0.1)
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
        #self.safe_write(':AWGC:RMOD ENH')
        self.num_of_lines = 0
        self.lmax = int(self.safe_query('WLISt:WAVeform:LMAX?'))
        self.lmin = int(self.safe_query('WLISt:WAVeform:LMIN?'))
        self.granularity = int(self.safe_query('WLISt:WAVeform:GRANularity?'))
        
        self.errors = [self.safe_query(':SYST:ERR?')]
        while not self.errors[-1].startswith('0'):
            self.errors.append(self.safe_query(':SYST:ERR?'))
        if len(self.errors) > 1 :
            print 'AWG ({addr}) has {num} error messages stored in self.errors.'.format(addr = self._addr, num = len(self.errors) -1)

        if 'dac_resolutions' in kwargs:
            self.dac_resolutions = kwargs['dac_resolutions']
        self.safe_write(':AWGC:RMOD CONT')
                
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
            if not 0.25 <= amps[i-1] <= 0.5:
                raise Exception('Amplitude out of range.')
            self.instr.write('SOUR%d:VOLT %f'%(i, amps[i-1]))
        
    @property
    def dac_resolutions(self):
        return [float(self.instr.query('SOUR%d:DAC:RES?'%i)) for i in [1,2]]

    @dac_resolutions.setter
    def dac_resolutions(self, resolutions):
        resol = make_iterable(resolutions, repeat_len = 2)
        for i in [1,2]:
            if not resol[i-1] in (8, 9, 10):
                raise Exception('Invalid number of bits.')
            self.instr.write('SOUR%d:DAC:RES %f'%(i, resol[i-1]))

    def _check_wfm_length(self, wfm_length, index):
        if not self.lmin <= wfm_length:
            print 'At step %d, the waveform length is %d.' %(index, wfm_length),
            raise Exception("The length of waveform for AWG70000 must be longer than %d."%self.lmin)
        if not wfm_length <= self.lmax:
            print 'At step %d, the waveform length is %d.' %(index, wfm_length),
            raise Exception("The length of waveform for AWG70000 must be shorter than %d."%self.lmax)
        if not wfm_length%self.granularity == 0:
            print 'At step %d, the waveform length is %d.' %(index, wfm_length),
            raise Exception("The length of waveform for AWG70000 must be dividable by %d" % self.granularity)
            
    def send_wfms(self, ch_id, **kwargs):
        if ('id_list' in kwargs and ch_id == kwargs['id_list'][0]) or kwargs.get('clear_buffer', False):
            self.safe_write('SLISt:SEQuence:DELete ALL')
            self.safe_write('WLISt:WAVeform:DELete ALL')
        id_list = kwargs.get('id_list', [ch_id,])
        if ch_id in self.ch1_ids() and ch_id == self.ch1_ids(id_list)[0]:
            bundle = waveform([self.ch1, self.mk11, self.mk12])
            suffix = '_ch1'
        elif ch_id in self.ch2_ids() and ch_id == self.ch2_ids(id_list)[0]:
            bundle = waveform([self.ch2, self.mk21, self.mk22])
            suffix = '_ch2'
        else:
            return
        
        chunk_size = 2**20
        wave_data, mk_data, len_list, name_list = bundle.format_TekWFM_AWG70000(chunk_size = chunk_size)
        
        for index, wfm_length in enumerate(len_list):
            self._check_wfm_length(wfm_length, index)
        
        self._OPC() #complete all pending operations
        for wave_block_chunked, mk_block_chunked, wfm_length, name in zip(wave_data, mk_data, len_list, name_list):
            info = ({'name': name + suffix, 'size': wfm_length})
            self.safe_write('WLISt:WAVeform:NEW "{name}",{size}'.format(**info))
            #self.safe_write('WLISt:WAVeform:NEW "{name}",{size},REAL'.format(**info))
            for chunk_index in range(len(wave_block_chunked)):
                numpnts = chunk_size if chunk_index < len(wave_block_chunked) -1 else wfm_length - chunk_index * chunk_size
                wave_block, mk_block = wave_block_chunked[chunk_index], mk_block_chunked[chunk_index]
                info.update({'start_index':chunk_index*chunk_size, 'size': numpnts})
                info.update({'wave_block_data':wave_block, 'marker_block_data':mk_block})
                self.instr.write_raw('WLISt:WAVeform:DATA "{name}",{start_index},{size},{wave_block_data}'.format(**info))
                self.instr.write_raw('WLISt:WAVeform:MARKer:DATA "{name}",{start_index},{size},{marker_block_data}'.format(**info))

                # cmd    = 'WLISt:WAVeform:DATA "{name}",{start_index},{size},{wave_block_data}'.format(**info)
                # header = 'WLISt:WAVeform:DATA "{name}",{start_index},{size},'.format(**info)
                # print cmd[:len(header)+10]
                # cmd    = 'WLISt:WAVeform:MARKer:DATA "{name}",{start_index},{size},{marker_block_data}'.format(**info)
                # header = 'WLISt:WAVeform:MARKer:DATA "{name}",{start_index},{size},'.format(**info)
                # print cmd[:len(header)+10]

                self._OPC()
    
    def load_seq(self, ch_id = 'ch1', **kwargs):
        self.stop()
        self.send_wfms(ch_id, **kwargs)
        id_list = kwargs.get('id_list', [ch_id,])
        
        if ch_id == id_list[-1]:
            self._OPC()#complete all pending operations
            main_ch =  major_channel([self.ch(ch) for ch in (self.ch1_ids() + self.ch2_ids())])         
            print 'Loading the sequence into AWG ({addr})'.format(addr=self._addr),
            start = time()
            for cmd in main_ch.seq.format_TekSCPI_AWG70000():
               self.safe_write(cmd)
               self._OPC()#complete all pending operations
            print ' in {time:.2f} seconds'.format(time=time() - start)
            self.num_of_lines = len(main_ch.seq.data)
            self.run()
    
    def run(self):
        self.safe_write(':AWGC:RUN:IMM')
        for n in [1, 2]:
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
