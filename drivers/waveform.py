# -*- coding: utf-8 -*-
"""

"""
from instr import container
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import chirp
import matplotlib.pyplot as plt
from math import ceil, fsum
#import pandas as pd
#import qgrid
from copy import deepcopy
from struct import pack
import re

def IEEE_block_format(block):
    #IEEE 488.2 definite length block format
    return '#' + str(len(str(len(block)))) + str(len(block)) + block

class sequence:
    #list of dictionaries
    def __init__(self, name = 'seq0'):
        self.data = []
        self.name = name
        self.last_comp = 0.
    
    def append(self, **kwargs):
        dictionary = {}
        dictionary['name'] = kwargs.get('name', '{name}_{index:03d}'.format(name = self.name, index = len(self.data)))
        dictionary['repeat'] = kwargs.get('repeat', 1)
        dictionary['repeat_0'] = int(round(dictionary['repeat'],0)) if not np.isinf(dictionary['repeat']) else 0
        dictionary['repeat_1'] = int(round(dictionary['repeat'],0)) if not np.isinf(dictionary['repeat']) else 1
        dictionary['wait'] = kwargs.get('wait', False)
        dictionary['go_to'] = kwargs.get('go_to', None)
        if kwargs.get('relative', False) and dictionary ['go_to'] is not None:
            dictionary['go_to'] += len(self.data)
        dictionary['target'] = kwargs.get('target', 0)
        dictionary['seq?'] = kwargs.get('seq?', False)
        dictionary['start'] = self.data[-1]['end'] if self.data else 0.
        dictionary['end'] = kwargs.get('end')
        self.data.append(dictionary)

    def undo_append(self, **kwargs):
        self.data.pop()

    def format_pandas(self):
        pass

    def interact(self):
        pass      
    
    def format_MAGIC3002(self, terminator = '\n', id_list = ['ch1', 'ch2']):
        line_list = []
        for dictionary in deepcopy(self.data):
            format_string = ''
            for ch_id in id_list:
                format_string += '"{name}.seq",' if dictionary['seq?'] else '"{name}_%s.wfm",'%(ch_id)
            format_string += '{repeat_0:.0f},{wait:.0f},{go_to:.0f},{target:.0f}'
            dictionary['go_to'] = dictionary['go_to']+1 if dictionary['go_to'] is not None else 0
            line_list.append(format_string.format(**dictionary))
        optional_info = ['JUMP_MODE SOFTWARE', 'JUMP_TIMING SYNC']
        return terminator.join(['MAGIC 3002','LINES %d'%len(line_list)] + line_list + optional_info)
    
    def format_TekSCPI(self, id_list = ['ch1', 'ch2']):
        commands = ['SEQ:LENG 0','SEQ:LENG %d'%len(self.data)]
        for n, dictionary in enumerate(deepcopy(self.data)):
            dictionary.update({'index': n+1})
            cmds = []
            if dictionary['seq?']:
                cmds = ['SEQ:ELEM{index}:SUBS "{name}"',]
            else:
                for ch_id in id_list:
                    cmds.append('SEQ:ELEM{index}:WAV%s "{name}_%s"'%(re.findall('^.*?([0-9]+)$',ch_id)[-1], ch_id))
            cmds.append('SEQ:ELEM{index}:LOOP:'+ ('INF 1' if np.isinf(dictionary['repeat']) else 'COUN {repeat:.0f}'))
            cmds.append('SEQ:ELEM{index}:TWA {wait:.0f}')
            if dictionary['go_to'] is not None:
                dictionary['go_to'] += 1
                cmds += ['SEQ:ELEM{index}:GOTO:IND {go_to:.0f}', 'SEQ:ELEM{index}:GOTO:STAT 1']
            commands += [cmd.format(**dictionary) for cmd in cmds]
        return commands	

def major_channel(wfm_ch_list):
    if len(wfm_ch_list) == 1:
        return wfm_ch_list[0]
    elif len(wfm_ch_list) == 2:
        data0, data1 = wfm_ch_list[0].seq.data, wfm_ch_list[1].seq.data
        if len(data0) == 0:
            return wfm_ch_list[1]
        elif len(data1) == 0:
            return wfm_ch_list[0]
        if len(data0) != len(data1):
            raise Exception('Waveforms are not compatible.')
        candidate_list = []
        for dict0, dict1 in zip(data0, data1):
            if dict0['repeat_0'] != dict1['repeat_0'] or dict0['seq?'] != dict1['seq?']:
                raise Exception('Waveforms are not compatible.')
            if dict0['start'] != dict1['start'] or dict0['end'] != dict1['end']:
                raise Exception('Waveforms are not compatible.')
            if dict0['wait'] != dict1['wait']:
                candidate_list.append(0 if dict0['wait'] else 1)
            if dict0['go_to'] is not None or dict1['go_to'] is not None:
                if dict0['go_to'] != dict1['go_to']:
                    raise Exception('Waveforms are not compatible.')
                candidate_list.append(0 if dict0['go_to'] is not None else 1)
        if len(candidate_list) == 0:
            return wfm_ch_list[0]
        if max(candidate_list) == min(candidate_list):
            return wfm_ch_list[candidate_list[0]]
        else:
            raise Exception('Waveforms are not compatible.')            
    else:
        return major_channel(wfm_ch_list[:-2] + [major_channel(wfm_ch_list[-2:]),])

class waveform_channel(object):
    def __init__(self, instr, **kwargs):
        self.instr = instr
        self.default_value = kwargs.get('default_value', 0.)
        self.default_frequency = kwargs.get('default_frequency', 0.)
        self.ch_list = [self,]
        self.ch_id = kwargs.get('ch_id', 0)
        self.name = ""
        self.scale = kwargs.get('scale', 1) # scale = 0.5 for a 6 dB loss in the line
        self.refresh()

    @property
    def t_sample(self):
        return self.instr.t_sample
    
    @t_sample.setter
    def t_sample(self, newval):
        self.instr.t_sample = newval

    @property
    def pulse_time(self):
        return self._pulse[-1][0] if self._pulse else 0.
        
    @pulse_time.setter
    def pulse_time(self, newval):
        if not newval == self.pulse_time:
            raise Exception('You are not supposed to change pulse_time.')
    
    def dwell(self, **kwargs):
        duration, pos, bur = self._check_inputs(**kwargs)
        if len(pos) == 0:#if the position is not specified
            pos = [self._pulse[-1][1] if len(self._pulse) else self.default_value]
        
        if len(self._pulse) > 0 and np.isnan(self._pulse[-1][1]):#previously ramping to nowhere
            self._pulse[-1] = (self._pulse[-1][0], pos[0])
        else:#the last position is given
            self._pulse.append((self.pulse_time, pos[0]))
        
        self._pulse.append((self.pulse_time + duration, pos[0]))
        self._phase = 'dwell'

    def ramp(self, **kwargs):
        if '_from' in kwargs:
            self.dwell(duration = 0, at = kwargs.pop('_from'))
        duration, pos, bur = self._check_inputs(**kwargs)
        
        if len(self._pulse) ==  0: #First segment
            self._pulse.append((0, self.default_value))
        elif np.isnan(self._pulse[-1][1]): #if the previous segment is also a ramp
            self._pulse[-1] = (self.pulse_time + duration, pos[0]) #make an unified ramp segment
        else:
            self._pulse.append((self.pulse_time + duration, pos[0] if pos else np.nan))
        self._phase = 'ramp'
    
    def excurse(self, **kwargs):
        duration, pos, bur = self._check_inputs(**kwargs)     
        self.dwell(duration = duration, at = pos[0])
        self.ramp(duration = 0., to = self.default_value)
    
    def compensate(self, **kwargs):
        duration, pos, bur = self._check_inputs(**kwargs)
        target = pos[0] if len(pos) == 1 else self.default_value
        
        if np.isnan(self._pulse[-1][1]):
            raise Exception("Cannot compensate while ramping to nowhere.")
               
        self.section(division = False, repeat = 1)
        seq_indx = [i for i, seq_dict in enumerate(self.seq.data) if seq_dict['end'] > self.seq.last_comp][0]
        tarr, wfm_list = self.processed_wave(start = self.seq.last_comp)
        wfm_weight = fsum([fsum(wfm)*seq_dict['repeat_0'] for wfm, seq_dict in zip(wfm_list, self.seq.data[seq_indx:])])
        self.seq.undo_append()
        
        cval = (self.time_global()* target - wfm_weight*float(self.t_sample))/duration
        self.dwell(duration = duration, at = cval)
        self._phase = 'compensated'
        self.seq.last_comp = self.pulse_time
        return cval
    
    def burst(self, **kwargs):
        if self._phase == 'ramp':
            raise Exception("Cannot burst while ramping to nowhere.")
        duration, pos, bur = self._check_inputs(**kwargs)
        if duration > 0.:
            amp, phase, freq, env = bur
            if np.isnan(amp) or np.isnan(freq):
                raise Exception('Amp and freq cannot be omitted.')
            
            self._burst.append(((self.pulse_time, self.pulse_time + duration), bur))
            if kwargs.get('auto_dwell', True):
                self.dwell(**kwargs)
            self._phase = 'burst'

    def time_global(self, pulse_time = None):
        pulse_time = self.pulse_time if pulse_time is None else pulse_time
        pre_secs   = [_dict for _dict in self.seq.data if _dict['end'] <= pulse_time]
        seq_gtime  = fsum([(_dict['end']-_dict['start'])*_dict['repeat_1'] for _dict in pre_secs])
        seq_ctime  = pre_secs[-1]['end'] if pre_secs else 0.
        return pulse_time - seq_ctime + seq_gtime

    def dividable(self):
        start = self.seq.data[-1]['end'] if self.seq.data else 0
        end = self.pulse_time
        degeneracy = max((0, len([True for t, val in self._pulse if t == start])-1))
        pulse_vals = [val for t, val in self._pulse if start <= t <= end][degeneracy:]
        burst_not_in_range = all([(end <= seg[0][0] or seg[0][1] <= start) for seg in self._burst])
        pulse_val_changes = len(pulse_vals) and max(pulse_vals) != min(pulse_vals)
        return not pulse_val_changes and burst_not_in_range
    
    def keep_up(self, time):
        to_go = time - self.time_global()
        if to_go > 0.:
            self.dwell(duration = to_go)
    
    def section(self,**kwargs):
        repeat = kwargs.pop('repeat', 1)
        start = self.seq.data[-1]['end'] if self.seq.data else 0
        end = self.pulse_time
        if start == end:
            return
        if kwargs.get('division', True if repeat == 1 else False) and self.dividable():
            degeneracy = len([True for t, val in self._pulse if t == start])
            pulse_vals = [val for t, val in self._pulse if start <= t <= end][degeneracy:]
            unit, rep = auto_division((end-start)/self.t_sample)
            if rep > 1:
                end = start + unit * self.t_sample
                repeat *= rep
                self._pulse = self._pulse[:-len(pulse_vals)]
                self._pulse.append((start, pulse_vals[0]))
                self._pulse.append((end, pulse_vals[0]))
        if start < self.seq.last_comp and self._phase != 'compensated':
            print 'Warning: the section continues after compensation.'
        self.seq.append(end = end, repeat = repeat, **kwargs)
    
    def refresh(self):
        self._pulse, self._burst = [], [] #Pulse is pulse, burst is burst.
        self.scaled_waveform_list, self.waveform_list, self.t_array_list = [], [], []
        self._phase, self.seq = 'new', sequence()
    
    def flatten_waves(self, scaled = False):
        wfm_flatten = np.zeros(0)
        wfm_list = self.waveform_list if scaled else self.scaled_waveform_list
        for wfm, seq_dict in zip(wfm_list, self.seq.data):
            wfm_flatten = np.append(wfm_flatten, [wfm]*seq_dict['repeat_1'])
        tarr_flatten = np.arange(0.5, len(wfm_flatten), 1.)*self.t_sample
        return tarr_flatten, wfm_flatten

    def compose(self, **kwargs):
        to_go = kwargs['time'] - self.time_global() if 'time' in kwargs else 0.
        if not self.seq.data or to_go > 0.:
            self.section(new = True, **kwargs)
        if np.isnan(self._pulse[-1][1]):
            raise Exception("Cannot compose while ramping to nowhere.")
        self.t_array_list, self.scaled_waveform_list = self.processed_wave(**kwargs)
        self.waveform_list = []
        for wfm in self.scaled_waveform_list:
            self.waveform_list.append(wfm/float(self.scale))
        self._phase = 'composed'
    
    def processed_wave(self, **kwargs):
        tarr_list, wfm_list = [], []
        arg_start, arg_end = kwargs.pop('start', 0.), kwargs.pop('end', np.inf)
        for _dict in self.seq.data:
            if arg_start < _dict['end'] and _dict['start'] < arg_end:
                start, end  = max(_dict['start'], arg_start), min(_dict['end'], arg_end)
                tarr, rawwave = self.raw_wave_concat(end = end, t_resolution = self.t_sample,
                                                     start = start, **kwargs)
                tarr_list.append(tarr)
                wave = rawwave #process here for further calibration, preamplification etc.
                wfm_list.append(wave)
        return tarr_list, wfm_list
    
    def _processed_wave(self, **kwargs):
        #depracted version to prevent memory error for very long pulses
        tarr, rawwave = self.raw_wave_concat(end = self.pulse_time, t_resolution = self.t_sample, **kwargs)
        wave = rawwave
        #process here for calibration, preamplification etc
        tarr_list, wfm_list = [], []
        for _dict in self.seq.data:
            start, end = _dict['start'], _dict['end']
            rng = np.logical_and(start <= tarr, tarr < end)
            tarr_list.append(tarr[rng])
            wfm_list.append(wave[rng])
        return tarr_list, wfm_list

    def raw_wave_concat(self, end, t_resolution, start = 0.):
        tarr = np.linspace(start, end, round((end-start)/t_resolution)+1.)[:-1]+0.5*t_resolution
        #Pulse is pulse.
        ts   = [segment[0] for segment in self._pulse]
        vals = [segment[1] for segment in self._pulse]
        pfunc = interp1d(ts, vals, bounds_error = False, assume_sorted = True)
        pulseraw = pfunc(tarr)
        
        #Burst is burst.
        burstraw = np.zeros_like(tarr)
        f_default = self.default_frequency
        for segment in self._burst:
            t0, t1 = segment[0]
            if start <= t0 and t1 <= end:
                t_shift = self.time_global(t0)-t0
                amp, phase, freq, env = segment[1]
                envarr = np.zeros_like(tarr)
                if env == "rec":
                    freq += f_default
                    envarr[np.argmin(np.abs(tarr-t0)):np.argmin(np.abs(tarr-t1))+1] = amp
                    burstraw += envarr*np.cos(2.*np.pi*freq*(tarr+t_shift) + phase)
                elif env in ("gauss", "deriv-gauss"):
                    freq += f_default
                    sig, center = (t1-t0)/4., (t1+t0)/2.
                    if env == "gauss":
                        envarr=amp*np.exp(-(tarr-center)**2./(2.*sig**2.))
                    elif env == "deriv-gauss":
                        envarr=amp*(-(tarr-center)/(sig))*np.exp(-(tarr-center)**2./(2.*sig**2.))
                    envarr[:np.argmin(np.abs((tarr-center)+2*sig))] = 0.
                    envarr[np.argmin(np.abs((tarr-center)-2.*sig))+1:] = 0.
                    burstraw += envarr*np.cos(2.*np.pi*freq*(tarr+t_shift) + phase)
                elif env == "chirp":
                    t = tarr[np.argmin(np.abs(tarr-t0)):np.argmin(np.abs(tarr-t1))]
                    osc = amp*chirp(t = t-t[0], t1 = t[-1]-t[0], f0 = f_default - 0.5*freq, f1 = f_default + 0.5*freq, phi = 180.*phase/np.pi)
                    pre, tail =np.zeros_like(tarr[:np.argmin(np.abs(tarr-t0))]), np.zeros_like(tarr[np.argmin(np.abs(tarr-t1)):])
                    burstraw += np.concatenate((pre, osc, tail))
            elif not (t1 <= start or end <= t0):
                raise Exception('Individual bursts have to be in a single waveform.')

        return tarr, pulseraw + burstraw
    
    def _check_inputs(self, **kwargs):
        pos_inputs, burst_inputs = [], []
        if kwargs['duration'] < 0:
            raise Exception("Duration cannot be negative.")

        for pos_key in ['at', 'in', 'to', '_from']:
            if pos_key in kwargs:
                if hasattr(kwargs[pos_key], '__iter__'):
                    raise Exception("More than one values are given to specify the single-ch output.")
                pos_inputs.append(kwargs[pos_key])
        if len(pos_inputs) > 1:
            raise Exception("Unable to interpret multiply specified positions.")
        
        for burst_key in ['amp', 'phase', 'freq', 'env']:
            if burst_key in kwargs:
                if hasattr(kwargs[burst_key], '__iter__'):
                    raise Exception("More than one values are given to specify the single-ch output.")                    
                burst_inputs.append(kwargs[burst_key])
            else:
                burst_inputs.append({'env':'rec', 'phase' :0}.get(burst_key, np.nan))
        
        return kwargs['duration'], pos_inputs, burst_inputs
    
    def send_wfms(self, **kwargs):
        self.instr.send_wfms(ch_id = self.ch_id, **kwargs)

    def load_seq(self, **kwargs):
        self.instr.load_seq(ch_id = self.ch_id, **kwargs)
    
    def t_arr_concat(self):
        return np.arange(0, self.pulse_time, self.t_sample)+0.5*self.t_sample

def make_iterable(inputs, repeat_len = 1):
    return inputs if hasattr(inputs, '__iter__') else [inputs]*repeat_len

def reshape(params):
    sorted_params = sorted([(k, np.asarray(param)) for k, param in enumerate(params)],
                           key = lambda p: len(p[1].shape), reverse = True)
    reshaped = [None]*len(params)
    if len(params) > 2:
        j = -1
        for k, param in sorted_params:
            reshaped[k] = param if j == -1 else reshape((reshaped[j], param))[1]
            j = k
    elif len(params) == 2:
        k_large, param_large = sorted_params[0]
        k_small, param_small = sorted_params[1]
        reshaped[k_large] = param_large
        dim_delta = len(param_large.shape) - len(param_small.shape)
        if dim_delta:
            extra = ((1,) if len(param_small.shape) > 0 else ())
            reshaped[k_small] = np.tile(param_small, param_large.shape[:dim_delta] + extra)
        else:
            reshaped[k_small] = param_small
        if not reshaped[0].shape == reshaped[1].shape:
            print reshaped[k_large].shape, reshaped[k_small].shape
            raise Exception('Too complicated to reshape properly')
    return reshaped

def auto_division(num, minimum = 1000):
    num = int(round(num,0))
    unit, _num = 1, num
    while _num%2 == 0 and unit < minimum:
        unit, _num = unit*2, _num/2
    if unit < minimum:
        _num, _minimum = int(round(num/unit,0)), int(ceil(float(minimum)/float(unit)))
        for n in range(_minimum, _num +1):
            if _num%n == 0:
                unit = n*unit
                break
    if unit < minimum:
        unit = num        
    return unit, num/unit
            
class waveform(object):
    def __init__(self, ch_list):
        self.ch_list = [ch for elem in ch_list for ch in elem.ch_list]
        for i, ch in enumerate(self.ch_list):
            if ch in self.ch_list[i+1:]:
                raise Exception("{Ch} is multiply used.".format(Ch = ch.name))

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.section(new = False)
    
    @property
    def t_sample(self):
        return [ch.t_sample for ch in self.ch_list]
    
    @t_sample.setter
    def t_sample(self, newval):
        for ch, val in zip(self.ch_list, make_iterable(newval, repeat_len = len(self.ch_list))):
            ch.t_sample = val

    @property
    def default_value(self):
        return [ch.default_value for ch in self.ch_list]
    
    @default_value.setter
    def default_value(self, newval):
        for ch, val in zip(self.ch_list, make_iterable(newval, repeat_len = len(self.ch_list))):
            ch.default_value = val

    @property
    def default_frequency(self):
        return [ch.default_frequency for ch in self.ch_list]
    
    @default_frequency.setter
    def default_frequency(self, newval):
        for ch, val in zip(self.ch_list, make_iterable(newval, repeat_len = len(self.ch_list))):
            ch.default_frequency = val

    def dwell(self, duration, **kwargs):
        for i, ch in enumerate(self.ch_list):
            ch.dwell(duration = duration, **self._ch_kwargs(i, **kwargs))

    def ramp(self, duration, **kwargs):
        for i, ch in enumerate(self.ch_list):
            ch.ramp(duration = duration, **self._ch_kwargs(i, **kwargs))

    def excurse(self, duration, **kwargs):
        for i, ch in enumerate(self.ch_list):
            ch.excurse(duration = duration, **self._ch_kwargs(i, **kwargs))
    
    def compensate(self, duration, **kwargs):
        return [ch.compensate(duration = duration, **self._ch_kwargs(i, **kwargs)) for i, ch in enumerate(self.ch_list)]
    
    def burst(self, duration, **kwargs):
        for i, ch in enumerate(self.ch_list):
            ch.burst(duration = duration, **self._ch_kwargs(i, **kwargs))

    def sync(self):
        latest = max([ch.time_global() for ch in self.ch_list])
        for ch in self.ch_list:
            ch.keep_up(time = latest)
    
    def section(self, **kwargs):
        self.sync()
        div = [kwargs.get('division', True), kwargs.get('repeat',1) == 1] + [ch.dividable() for ch in self.ch_list]
        kwargs['division'] = min(div) #True only if all conditions are met
        for ch in self.ch_list:
            ch.section(**kwargs)
    
    def refresh(self):
        for ch in self.ch_list:
            ch.refresh()
    
    def compose(self):
        self.section(new = False)
        
        for ch in self.ch_list:
            ch.compose() #compose each ch first
            
        length_list =  [ch.time_global() for ch in self.ch_list]
        if max(length_list) != min(length_list):
            print length_list
            raise Exception("Waveform lengths are different.")
    
    def send_wfms(self, **kwargs):
        for ch in self.ch_list:
            kwargs['id_list'] = [c2.ch_id for c2 in self.ch_list if c2.instr == ch.instr]
            ch.send_wfms(**kwargs)
    
    def load_seq(self, **kwargs):
        for ch in self.ch_list:
            kwargs['id_list'] = [c2.ch_id for c2 in self.ch_list if c2.instr == ch.instr]
            ch.load_seq(**kwargs)

    def _ch_kwargs(self, ch_num, **kwargs):
        ch_kw = kwargs
        for key in kwargs:
            if key == 'duration':
                if kwargs['duration'] < 0.:
                    raise Exception("Duration cannot be negative.") 
            else:
                kwargs[key] = make_iterable(kwargs[key], repeat_len = len(self.ch_list))
                if len(kwargs[key]) != len(self.ch_list):
                    raise Exception("%s must contain %d points."%(key,len(self.ch_list)))
                ch_kw[key] = kwargs[key][ch_num]
        return ch_kw
    
    def show(self, **kwargs):
        fig, axarr = plt.subplots(len(self.ch_list), sharex=True, figsize = kwargs.get('figsize', None))
        axarr = [axarr,] if len(self.ch_list) == 1 else axarr
        mode = 'stack' if not kwargs.get('flatten', False) else 'flatten'
        
        for i, ch in enumerate(self.ch_list):
            wfm_list = ch.scaled_waveform_list if kwargs.get('scaled', True) else ch.waveform_list
            ymax = max([max(wfm) for wfm in wfm_list])
            ymin = min([min(wfm) for wfm in wfm_list])
            ypos = ymax + 0.1* (ymax - ymin)
            if mode == 'stack':
                for tarr, wfm, seq_dict in zip(ch.t_array_list, wfm_list, ch.seq.data):
                    t = np.insert(tarr, [0, len(tarr)], [tarr[0]-0.5*ch.t_sample, tarr[-1]+0.5*ch.t_sample])
                    w = np.insert(wfm, [0, len(wfm)], [wfm[0], wfm[-1]])
                    axarr[i].step(t, w, where = 'mid')
                    axarr[i].axvline(x = t[-1], color = 'k', alpha = 1. if np.isinf(seq_dict['repeat']) else 0.5)
                    exp = 'x {repeat}'.format(**seq_dict)
                    axarr[i].text(x = (t[0] + t[-1])/2., y = ypos, s = exp, ha = 'center', va = 'top' )

            if mode == 'flatten':
                tarr_flatten, wfm_flatten = ch.flatten_waves(scaled = kwargs.get('scaled', False))
                axarr[i].step(tarr_flatten, wfm_flatten)
                time_global = 0.
                for seq_dict in ch.seq.data:
                    for j in range(int(seq_dict['repeat_1'])):
                        duration = seq_dict['end'] - seq_dict['start']
                        axarr[i].axvline(x = time_global, color = 'k', alpha = 0.5)
                        time_global += duration
            
            if not ymax == ymin:
                axarr[i].set_ylim([ymin - 0.1* (ymax - ymin), ymax + 0.1* (ymax - ymin)])
            axarr[i].set_ylabel(ch.name)
        try:
            fig.patch.set_alpha(1.0);fig.patch.set_facecolor('w');plt.tight_layout()
        except:
            pass

    def format_MAGIC1000(self):
        #Tektronix AWGs
        main = major_channel(self.ch_list)
        ch, mk1, mk2 = self.ch_list
        defaults = ch.default_value, mk1.default_value, mk2.default_value
        magic_file_list, len_list, name_list = [], [], []
        
        for n, main_wfm in enumerate(main.waveform_list):
            ch_wfm = ch.waveform_list[n] if len(ch.waveform_list) > n else np.zeros(0)
            mk1_wfm = mk1.waveform_list[n] if len(mk1.waveform_list) > n else np.zeros(0)
            mk2_wfm = mk2.waveform_list[n] if len(mk2.waveform_list) > n else np.zeros(0)
        
            ch_wfm = np.append(ch_wfm, defaults[0]*np.ones(len(main_wfm)-len(ch_wfm)))
            mk1_wfm = np.clip(np.append(mk1_wfm, defaults[1]*np.ones(len(main_wfm)-len(mk1_wfm))), 0., 1.)
            mk2_wfm = np.clip(np.append(mk2_wfm, defaults[2]*np.ones(len(main_wfm)-len(mk2_wfm))), 0., 1.)
                    
            if min(ch_wfm) < -1. or 1. < max(ch_wfm):
                raise Exception('Output out of range.')
            trailer = ('CLOCK %13.10e\n' % (1e+9/main.t_sample)).replace("+","")
            data = ''
            for p in range(len(ch_wfm)):
                w, m1, m2 = ch_wfm[p], mk1_wfm[p], mk2_wfm[p]
                data += pack('<fB', w, int(round(m1+2*m2,0)))
            magic_file_list.append('MAGIC 1000\n' + IEEE_block_format(data) + trailer)
            len_list.append(len(main_wfm))
            name_list.append(main.seq.data[n]['name'])
        
        return magic_file_list, len_list, name_list

    def format_TekWFM(self, format = 'real'):
        #Tektronix AWGs 5000 and 7000 series
        main = major_channel(self.ch_list)
        ch, mk1, mk2 = self.ch_list
        defaults = ch.default_value, mk1.default_value, mk2.default_value
        file_list, len_list, name_list = [], [], []
        
        for n, main_wfm in enumerate(main.waveform_list):
            ch_wfm = ch.waveform_list[n] if len(ch.waveform_list) > n else np.zeros(0)
            mk1_wfm = mk1.waveform_list[n] if len(mk1.waveform_list) > n else np.zeros(0)
            mk2_wfm = mk2.waveform_list[n] if len(mk2.waveform_list) > n else np.zeros(0)
        
            ch_wfm = np.append(ch_wfm, defaults[0]*np.ones(len(main_wfm)-len(ch_wfm)))
            mk1_wfm = np.clip(np.append(mk1_wfm, defaults[1]*np.ones(len(main_wfm)-len(mk1_wfm))), 0., 1.)
            mk2_wfm = np.clip(np.append(mk2_wfm, defaults[2]*np.ones(len(main_wfm)-len(mk2_wfm))), 0., 1.)
                    
            if min(ch_wfm) < -1. or 1. < max(ch_wfm):
                raise Exception('Output out of range.')
            data = ''
            wvmk = np.clip((ch_wfm+1.)*(2**13)-1., 0, 2**14-1)+ (mk1_wfm+2*mk2_wfm)*(2**14)
            for p in wvmk:
                data += pack('<h', p)
            file_list.append(IEEE_block_format(data))
            len_list.append(len(main_wfm))
            name_list.append(main.seq.data[n]['name'])
        
        return file_list, len_list, name_list

class AWG_instr(container):  
    def __setattr__(self, name, value):
        if isinstance(value, waveform_channel) and not value.name:
            value.name = self.name + '.' + name if self.name else name
        super(container, self).__setattr__(name, value)

    def _selfinit(self, **kwargs):
        if 'no_visa' in kwargs: #For debugging and as an example.
            self.no_visa = True
            self.t_sample = kwargs.get('t_sample', 1.)
            self.catalog_seq, self.catalog_wfm = {}, {}
            self.ch1 = waveform_channel(instr = self, ch_id = 'ch1')
            self.ch2 = waveform_channel(instr = self, ch_id = 'ch2')
            self.mk11 = waveform_channel(instr = self, ch_id = 'mk11')
            self.mk12 = waveform_channel(instr = self, ch_id = 'mk12')
            self.mk21 = waveform_channel(instr = self, ch_id = 'mk21')
            self.mk22 = waveform_channel(instr = self, ch_id = 'mk22')
        else:
            raise NotImplementedError("You must override _selfinit function.")
        #This function is instrument dependent.
        #Override this method for a driver
        #Include self.t_sample = kwargs.get('t_sample', 1.) or the like
        #Define each waveform or marker channel as waveform_channel
        
    @property
    def t_sample(self):
        if hasattr(self, 'no_visa') and self.no_visa:
            return self._t_sample
        else:
            raise NotImplementedError("You must override t_sample property.")

    @t_sample.setter
    def t_sample(self, newval):
        if hasattr(self, 'no_visa') and self.no_visa:
            self._t_sample = newval
        else:
            raise NotImplementedError("You must override t_sample property.")

    def load_seq(self, ch_id, id_list, **kwargs):
        if not hasattr(self, 'no_visa') or self.no_visa == False:
            raise NotImplementedError("You must override load_seq function.")
