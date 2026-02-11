#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
*FACADE PATTER - What is this pattern about?
The Facade pattern is a way to provide a simpler unified interface to
a more complex system. It provides an easier way to access functions
of the underlying system by providing a single entry point.

QFacade is initialized with inner classes, client users only need
to call them respectively.
*References:
https://sourcemaking.com/design_patterns/facade
https://fkromer.github.io/python-pattern-references/design/#facade
http://python-3-patterns-idioms-test.readthedocs.io/en/latest/ChangeInterface.html#facade

Provides a simpler unified interface to a complex system.
"""

from __future__ import print_function
import time
from .manual_decline.arpsfcn import arpsfcn
from .manual_decline.thbfcn import thbfcn
from .manual_decline.fitarps import fitarps
from .manual_decline.fitthb import fitthb
from .manual_decline.thb2arps import thb2arps

# Complex and inherited from manual decline
''' keep adding classes to be in sync with Brennan changes...
    Pattern is good when we need to change the inner then client use
    wont need to know...
'''
# Facade Patterns
class QFacade(object):

    def __init__(self):

        self.fitarps = fitarps
        self.thbfcn = thbfcn
        self.fitthb = fitthb
        self.arpsfcn = arpsfcn



    def run_fit_arps(self,**kwargs):
        fitarpsObj = self.fitarps
        return fitarpsObj


    def run_thb_fcn(self,**kwargs):
        thbfcnObj = self.thbfcn
        return thbfcnObj

    def run_fit_thb(self,**kwargs):
        fitthbObj = self.fitthb
        return fitthbObj

    def run_arps_fcn(self,**kwargs):
        arpsObj = self.arpsfcn
        return arpsObj

    def run_thb2_arps(self, **kwargs):
        thb2arpsObj = thb2arps
        return thb2arpsObj
