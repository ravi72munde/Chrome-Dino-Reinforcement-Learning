
import os
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
game_url = "game/dino.html"
chrome_driver_path = "../chromedriver.exe"
class Game:
    def __init__(self,custom_config=True):
        chrome_options = Options()
        chrome_options.add_argument("disable-infobars")
        self._driver = webdriver.Chrome(executable_path = chrome_driver_path,chrome_options=chrome_options)
        self._driver.set_window_position(x=-10,y=0)
        self._driver.set_window_size(200, 300)
        self._driver.get(os.path.abspath(game_url))
        if custom_config:
            self._driver.execute_script("Runner.config.ACCELERATION=0")
    def get_crashed(self):
        return self._driver.execute_script("return Runner.instance_.crashed")
    def get_playing(self):
        return self._driver.execute_script("return Runner.instance_.playing")
    def restart(self):
        self._driver.execute_script("Runner.instance_.config.CLEAR_TIME = "+ str(randint(1, 999)))
        self._driver.execute_script("Runner.instance_.restart()")
#         self._driver.refresh()
#         self.press_up()
        time.sleep(0.25)
    def press_up(self):
        self._driver.find_element_by_tag_name("body").send_keys(Keys.ARROW_UP)
    def press_down(self):
        self._driver.find_element_by_tag_name("body").send_keys(Keys.ARROW_DOWN)
    def get_score(self):
        score_array = self._driver.execute_script("return Runner.instance_.distanceMeter.digits")
        score = ''.join(score_array)
        return int(score)
    def pause(self):
        return self._driver.execute_script("return Runner.instance_.stop()")
    def resume(self):
        return self._driver.execute_script("return Runner.instance_.play()")
    def end(self):
        self._driver.close()

class Game_sate:
    def __init__(self,agent,game):
        self._agent = agent
        self._game = game
        self._display = show_img()
        self._display.__next__()
    def get_state(self,actions):
        actions_df.loc[len(actions_df)] = actions[1]
        reward = 0.1
        is_over = False
        if actions[1] == 1:
            self._agent.jump()

        image = grab_screen()
        self._display.send(image)

        if self._agent.is_crashed():
            scores_df.loc[len(loss_df)] = self._game.get_score()
            self._game.restart()
            reward = -5
            is_over = True
        return image, reward, is_over


