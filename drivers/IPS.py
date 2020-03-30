# -*- coding: utf-8 -*-
"""

Driver for IPS Superconducting Magnet Power Supply
"""
from instr import container, variable
import time


class IPS(container):

    def _selfinit(self, **kwargs):
        self.instr.timeout = 200
        self.instr.write_termination = '\r'
        self.instr.read_termination = '\r'

        self.safe_write("$C3") # Remote and unlock, suppressing echo
        self.safe_write("$A0") # Hold, suppressing echo
        self.safe_write("Q4") # Extended resolution

    @variable(type=float, min=-5, max=5, sweep_rate=0.1, units='T')
    def field(self):
        return self.output_field if self.switch_heater else self.persistent_field

    @field.setter
    def field(self, setval):
        self.switch_heater = True
        self.safe_write("$J%.4f" % setval)
        self.safe_write("$A1")
        try:
            while True:
                if abs(setval - self.output_field) < 1e-4:
                    break
        except KeyboardInterrupt:
            # Abort field ramp
            self.safe_write("$A0")

    @field.sweep_rate
    def field_sweep_rate(self):
        if not hasattr(self, '_sweep_rate'):
            self._sweep_rate = float(self.safe_query("R9")[1:])
        return self._sweep_rate

    @field_sweep_rate.setter
    def field_sweep_rate(self, setval):
        self._sweep_rate = float(setval)
        self.safe_write("$T%.4f" % setval)

    @field.switch_heater
    def field_switch_heater(self):
        heater_status = int(self.safe_query("X")[8:9])
        if heater_status == 1:
            switch = True
        elif heater_status in (0,2,8):
            switch = False
        else:
            raise Exception("Heater Fault with status %d (heater is on but current is low)" % heater_status)
        return switch

    @field_switch_heater.setter
    def field_switch_heater(self, setval):
        heater_status = self.switch_heater
        if heater_status == setval:
            return

        if setval: # Turn on switch heater
            field_in_magnet = self.persistent_field
            self.safe_write("$J%.4f" % field_in_magnet)
            self.safe_write("$A1") # To Set Point
            print("Heater on. Please wait for a moment until the power supply current reaches the persistent current in the magnet.")
            while True:
                if abs(field_in_magnet - self.output_field) < 1e-4:
                    break
                time.sleep(1.0)
            time.sleep(3.0)
            self.safe_write("$H1")
            time.sleep(20.0)
        else: # Turn off switch heater
            self.safe_write("$H0")
            print("Heater off. Please wait for 20 sec.")
            time.sleep(20.0)

    @field.persistent
    def field_persistent(self):
        return not self.switch_heater

    @field_persistent.setter
    def field_persistent(self, setval):
        persistent_status = not self.switch_heater
        if persistent_status == setval:
            return

        if setval:
            time.sleep(3.0)
            self.switch_heater = False
            self.safe_query("A2") # Go to Zero
            while True:
                if abs(self.output_field) < 1e-4:
                    break
                time.sleep(1.0)
            self.safe_query("A0") # Hold
        else:
            self.switch_heater = True

    @field.output_field
    def field_output_field(self):
        return float(self.safe_query("R7")[1:])

    @field.persistent_field
    def field_persistent_field(self):
        return float(self.safe_query("R18")[1:])
