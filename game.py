# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 15:32:51 2018

@author: ravi7
"""

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
class Game:
    def __init__(self):
        chrome_options = Options()
        chrome_options.add_argument("disable-infobars")
        self._driver = webdriver.Chrome(executable_path = "../chromedriver.exe",chrome_options=chrome_options)
        self._driver.set_window_position(x=-10,y=0)
        self._driver.set_window_size(200, 300)
        self._driver.get("chrome://dino")
    def press_up(self):
        self._driver.find_element_by_tag_name("body").send_keys(Keys.ARROW_UP)
    def press_down(self):
        self._game._driver.find_element_by_tag_name("body").send_keys(Keys.ARROW_DOWN)
    def end(self):
        self._driver.close()