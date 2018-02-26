# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 15:34:05 2018

@author: ravi7
"""

class DinoAgent:
    def __init__(self,game):
        self._game = game;
    def jump(self):
        self._game.press_up()
    def duck(self):
        self._game.press_down()
        