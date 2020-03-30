# -*- coding: utf-8 -*-
"""

"""

#from waveform import AWG_instr, waveform_channel, waveform
#from waveform import major_channel, make_iterable, IEEE_block_format
from waveform_lazy import AWG_instr, waveform_channel, waveform
from waveform_lazy import major_channel, make_iterable
import numpy as np
from time import time, sleep
import sys
from instr import KeyboardInterruptProtection

from Queue import Queue
from threading import Thread


def print_status(msg, overwrite=True):
    # print a one-line, overwritten message
    sys.stdout.write("\r" + msg if overwrite else msg)
    sys.stdout.flush()


class AWG5000(AWG_instr):
    """
    addr: For TCPIP, specify host addr (e.g. '192.168.1.2') with interface = 'LAN' and 
          start the VXI-11 server in the AWG 5000 instrument.
    """

    def _selfinit(self, **kwargs):
        #super(AWG5000, self)._selfinit(**kwargs)
        if hasattr(self, 'instr'):
            self.instr.read_termination = '\n'
            self.instr.timeout = 30000  #30 sec: some responses might be very slow
            self.clear_buffer()

            self.t_sample = kwargs.get('t_sample', 0.1)
            self.catalog_seq, self.catalog_wfm = {}, {}
            self.safe_write(':AWGC:RMOD SEQ')

            # Clear error queue
            self.check_errors(raise_exception=False)
        
        self.ch1 = waveform_channel(instr = self, ch_id = 'ch1')
        self.ch2 = waveform_channel(instr = self, ch_id = 'ch2')
        self.mk11 = waveform_channel(instr = self, ch_id = 'mk11')
        self.mk12 = waveform_channel(instr = self, ch_id = 'mk12')
        self.mk21 = waveform_channel(instr = self, ch_id = 'mk21')
        self.mk22 = waveform_channel(instr = self, ch_id = 'mk22')
        self.ch3 = waveform_channel(instr = self, ch_id = 'ch3')
        self.ch4 = waveform_channel(instr = self, ch_id = 'ch4')
        self.mk31 = waveform_channel(instr = self, ch_id = 'mk31')
        self.mk32 = waveform_channel(instr = self, ch_id = 'mk32')
        self.mk41 = waveform_channel(instr = self, ch_id = 'mk41')
        self.mk42 = waveform_channel(instr = self, ch_id = 'mk42')

        if hasattr(self, 'instr'):
            self.software_sequencer = kwargs.get('software_sequencer', False)
            self.safe_write(':AWGC:RMOD ENH')
            self.num_of_lines = 0
            self.scales = kwargs.get('scales', 1.)
            self.amplitudes = kwargs.get('amplitudes', 1.0)
    
    @property
    def t_sample(self):
        return  self._t_sample if hasattr(self,'instr') else 1.0

    @t_sample.setter
    def t_sample(self, newval):
        self._t_sample = np.clip(newval, 1./1.2, 100)
        self.safe_write(":FREQ %13.10f MHz" % (1000./self._t_sample))

    @property
    def scales(self):
        return [self.ch1.scale, self.ch2.scale,self.ch3.scale,self.ch4.scale]
    
    @scales.setter
    def scales(self, scales):
        scales = make_iterable(scales, repeat_len = 4)
        self.ch1.scale, self.ch2.scale,self.ch3.scale,self.ch4.scale = scales[0], scales[1],scales[2],scales[3]
    
    @property
    def amplitudes(self):
        return [float(self.safe_query('SOUR%d:VOLT?'%i)) for i in [1,2,3,4]]

    @amplitudes.setter
    def amplitudes(self, amps):
        amps = make_iterable(amps, repeat_len = 4)
        for i in [1,2,3,4]:
            if not 0.02 <= amps[i-1] <= 4.5:
                raise Exception('Amplitude out of range.')
            self.safe_write('SOUR%d:VOLT %f'%(i, amps[i-1]))

#"""  
#    @property
#    def dac_resolutions(self):
#        return [float(self.instr.query('SOUR%d:DAC:RES?'%i)) for i in [1,2,3,4]]
#
#    @dac_resolutions.setter
#    def dac_resolutions(self, resolutions):
#        resol = make_iterable(resolutions, repeat_len = 4)
#        for i in [1,2,3,4]:
#            if not resol[i-1] in (8, 10):
#                raise Exception('Invalid number of bits.')
#            self.instr.write('SOUR%d:DAC:RES %f'%(i, resol[i-1]))
#"""

    def _check_wfm_length(self, wfm_length, index):
        if wfm_length < 256:
            raise Exception("The length of waveform for AWG5000 must be no less than 256 for correct sequencer operation.")

    def ch1_ids(self, id_list = ['ch1', 'mk11', 'mk12']):
        return [_id for _id in id_list if _id in ['ch1', 'mk11', 'mk12']]
    
    def ch2_ids(self, id_list = ['ch2', 'mk21', 'mk22']):
        return [_id for _id in id_list if _id in ['ch2', 'mk21', 'mk22']]

    def ch3_ids(self, id_list = ['ch3', 'mk31', 'mk32']):
        return [_id for _id in id_list if _id in ['ch3', 'mk31', 'mk32']]
    
    def ch4_ids(self, id_list = ['ch4', 'mk41', 'mk42']):
        return [_id for _id in id_list if _id in ['ch4', 'mk41', 'mk42']]       
    
    def ch(self, ch_id):
        if ch_id in self.ch1_ids() or ch_id in self.ch2_ids() or ch_id in self.ch3_ids() or ch_id in self.ch4_ids():
            return getattr(self, ch_id)

    def send_wfms(self, ch_id, **kwargs):
        if ('id_list' in kwargs and ch_id == kwargs['id_list'][0]) or kwargs.get('clear_buffer', False):
            self.stop()
            print_status("Deleting waveforms...")
            self.safe_write('WLISt:WAVeform:DELete ALL')
        id_list = kwargs.get('id_list', [ch_id,])
        if ch_id in self.ch1_ids() and ch_id == self.ch1_ids(id_list)[0]:
            bundle = waveform([self.ch1, self.mk11, self.mk12])
            suffix = '_ch1'
        elif ch_id in self.ch2_ids() and ch_id == self.ch2_ids(id_list)[0]:
            bundle = waveform([self.ch2, self.mk21, self.mk22])
            suffix = '_ch2'
        elif ch_id in self.ch3_ids() and ch_id == self.ch3_ids(id_list)[0]:
            bundle = waveform([self.ch3, self.mk31, self.mk32])
            suffix = '_ch3'
        elif ch_id in self.ch4_ids() and ch_id == self.ch4_ids(id_list)[0]:
            bundle = waveform([self.ch4, self.mk41, self.mk42])
            suffix = '_ch4'
        else:
            return
        
        for index, wfm_length in enumerate(bundle.wfm_len_list()):
            self._check_wfm_length(wfm_length, index)

        self._OPC() #complete all pending operations
        self.check_errors()

        wfm_queue = Queue(maxsize=2)
        def _gen_wfms():  # Generate waveform text asynchronously
            for wfm_file, wfm_length, name in bundle.gen_format_TekWFM(format='integer'):
                info = ({'name': name + suffix, 'start_index':0, 'size': wfm_length, 'block_data': wfm_file})
                wfm_queue.put(info, block=True)
            wfm_queue.put(None, block=True)  # End flag
        t = Thread(target=_gen_wfms)
        t.daemon = True
        t.start()

        start = time()
        send_time = 0.
        sent_names = []
        while True:
            info = wfm_queue.get(block=True, timeout=30.0)
            if info is None:
                break
            if info['name'] not in sent_names:
                print_status("Sending {name}...".format(**info))
                from_time = time()
    #            self.safe_write('WLISt:WAVeform:NEW "{name}",{size},REAL'.format(**info))
                self.safe_write('WLISt:WAVeform:NEW "{name}",{size},INTeger'.format(**info))
                self._OPC()
                with KeyboardInterruptProtection():
                    self.instr.write_raw('WLISt:WAVeform:DATA "{name}",{start_index},{size},{block_data}'.format(**info))
                send_time += time() - from_time
                sent_names.append(info['name'])
        print_status("Waveforms generated in {gen_time:.2f} and sent in {time:.2f}".format(time=send_time, gen_time=time() - start - send_time))
        self.check_errors()
    
    def load_seq(self, ch_id='ch1', **kwargs):
        self.send_wfms(ch_id, **kwargs)
        id_list = kwargs.get('id_list', [ch_id,])
        id_list = [ch for ch in id_list if ch.startswith('ch')]
        
        if ch_id == id_list[-1]:
            self._OPC()  #complete all pending operations
            main_ch =  major_channel([self.ch(ch) for ch in (self.ch1_ids() + self.ch2_ids() + self.ch3_ids() + self.ch4_ids())])
            print_status('Loading the sequence into AWG ({addr})'.format(addr=self._addr))
            start = time()
            for cmd in main_ch.seq.format_TekSCPI(id_list=id_list):
                self.safe_write(cmd)
                self._OPC()  #complete all pending operations
            print_status(' in {time:.2f} seconds'.format(time = time() - start), overwrite=False)
            self.num_of_lines = len(main_ch.seq.data)

            self.check_errors()
            self.run(id_list=id_list, jump=kwargs.get('jump', None))
    
    def run(self, id_list=['ch1', 'ch2', 'ch3', 'ch4'], jump=None):
        # AWG must be run *before* channel outputs are turned on
        # to prevent unwanted waveform output
        self.safe_write(':AWGC:RUN:IMM')
        print_status("Starting AWG sequence. This may take a moment...")
        counter = 0
        while self.safe_query(':AWGC:RST?') not in (u'1', u'2'):
            counter += 1
            sleep(0.1)
            if counter % 10 == 0:
                self.check_errors()
        if jump is not None:
            self.jump(jump)
        print_status("Turning AWG output on. This may take a moment...")
        for ch in id_list:
            n = int(ch.lstrip('ch'))
            self.safe_write(':OUTP%d:STAT ON' % (n))
            counter = 0
            while self.safe_query(':OUTP%d:STAT?' % (n)) != u'1':
                counter += 1
                sleep(0.1)
                if counter % 10 == 0:
                    self.check_errors()
        print_status("Done.")
        self.check_errors()

    def stop(self):
        # All channel outputs must be turned off *before* AWG is stopped
        # to prevent unwanted waveform output
        print_status("Turning AWG output off. This may take a moment...")
        for n in [1, 2, 3, 4]:
            self.safe_write(':OUTP%d:STAT OFF' % (n))
            counter = 0
            while self.safe_query(':OUTP%d:STAT?' % (n)) != u'0':
                counter += 1
                sleep(0.1)
                if counter % 10 == 0:
                    self.check_errors()
        self.safe_write(':AWGC:STOP:IMM')
        print_status("Stopping AWG sequence. This may take a moment...")
        counter = 0
        while self.safe_query(':AWGC:RST?') in (u'1', u'2'):
            counter += 1
            sleep(0.1)
            if counter % 10 == 0:
                self.check_errors()
        print_status("Done.")
        self.check_errors()
        
    def jump(self, to, ch_id = 'ch1', **kwargs):
        id_list = kwargs.get('id_list', [ch_id,])
        if ch_id == id_list[-1]:
            t0 = time()
            while self.safe_query(':AWGC:RST?') != u'2':
                if time() > t0 + 10:#wait up to 10 seconds
                    raise Exception('Cannot jump while the AWG is waiting for trigger or stopped.')
            self.safe_write('SEQuence:JUMP:IMMediate %d'%(to%self.num_of_lines+1))

    def trigger(self):
        self.safe_write('*TRG')

    def _OPC(self):
        orig_timeout, self.instr.timeout = self.instr.timeout, float('+inf')
        try:
            code = self.instr.query('*OPC?')  # Wait forever, accepting keyboard interrupt
            if code == u'1':
                return
            else:
                print("*OPC? returned an unexpected code '{0}'".format(code))
                raise
        except:
            raise Exception("Failed to complete pending operations. You may have to call clear_buffer() to clear the message buffer.")
        finally:
            self.instr.timeout = orig_timeout

    def check_errors(self, raise_exception=True):
        self.errors = []
        while True:
            error = self.safe_query(':SYST:ERR?')
            if error.startswith('0'):
                break
            else:
                self.errors.append(error)
        if self.errors:
            if raise_exception:
                msg = 'Error occurred in AWG5000 (@ {addr}):\n'.format(addr=self._addr)
                for e in self.errors:
                    msg += e.replace('\r', '') + '\n'
                raise Exception(msg)
            else:
                print 'AWG ({addr}) has {num} error messages stored in self.errors.'.format(addr=self._addr,
                                                                                        num=len(self.errors))
