import threading
import time

import numpy as np

from pysc2 import maps
from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features
from pysc2.env import available_actions_printer
from pysc2.env import run_loop
from pysc2.env import sc2_env
from pysc2.lib import stopwatch
from pysc2.env.environment import StepType

from absl import app

# Options
MAP ='Simple64'
RACE = 'T'
OPPONENT_RACE = 'T'
DIFFICULTY = '1'

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_NO_OP = actions.FUNCTIONS.no_op.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_BUILD_BARRACKS = actions.FUNCTIONS.Build_Barracks_screen.id
_BUILD_SUPPLYDEPOT = actions.FUNCTIONS.Build_SupplyDepot_screen.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_TRAIN_MARINE = actions.FUNCTIONS.Train_Marine_quick.id
_ATTACK_MINIMAP = actions.FUNCTIONS.Attack_minimap.id
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index

# Unit IDs
_TERRAN_BARRACKS = 21
_TERRAN_COMMANDCENTER = 18
_TERRAN_SUPPLYDEPOT = 19
_TERRAN_SCV = 45

# Parameters
_PLAYER_SELF = 1
_SUPPLY_USED = 3
_SUPPLY_MAX = 4
_SCREEN = [0]
_MINIMAP = [1]
_QUEUED = [1]
_NOADD = [0]

def parse_obs():
    pass

def get_units(obs, _unit_type):
    unit_type = obs.observation["screen"][_UNIT_TYPE]
    unit_y, unit_x = (unit_type == _unit_type).nonzero()

    return (unit_y, unit_x)

def determine_base_location(obs):
    player_y, player_x = (obs.observation["minimap"][_PLAYER_RELATIVE] == _PLAYER_SELF).nonzero()
    return player_y.mean() <= 31

def action_is_valid(obs, action):
    return action in obs.observation["available_actions"]

def at_max_supply(obs):
    return not obs.observation["player"][_SUPPLY_USED] < obs.observation["player"][_SUPPLY_MAX]

class HelloWorld(base_agent.BaseAgent):
    base_top_left = None
    supply_depot_built = False
    scv_selected = False
    barracks_built = False
    barracks_selected = False
    army_selected = False
    
    def initialize(self, obs):
        self.base_top_left = determine_base_location(obs)
    
    def step(self, obs):
        super(HelloWorld, self).step(obs)
        
        time.sleep(0.1)
        
        if obs.step_type == StepType.FIRST:
            self.initialize(obs)
        
        self.supply_depot_built = get_units(obs, _TERRAN_SUPPLYDEPOT)[0].any()
        self.barracks_built = get_units(obs, _TERRAN_BARRACKS)[0].any()
        
        if not self.supply_depot_built:
            if not self.scv_selected:
                # Select a worker
                unit_y, unit_x = get_units(obs, _TERRAN_SCV)
                
                # Just need one, doesn't matter which
                target = [unit_x[0], unit_y[0]]
                
                self.scv_selected = True

                return actions.FunctionCall(_SELECT_POINT, [_SCREEN, target])
            
            elif action_is_valid(obs, _BUILD_SUPPLYDEPOT):

                unit_y, unit_x = get_units(obs, _TERRAN_COMMANDCENTER)
                
                # Try randomly around the area until space is found
                offset_x = np.random.randint(-4, 4)
                offset_y = np.random.randint(-4, 4)
                
                target = (np.min(unit_x)+offset_x, np.min(unit_y)+offset_y)
                
                return actions.FunctionCall(_BUILD_SUPPLYDEPOT, [_SCREEN, target])
            
        elif not self.barracks_built:
            if action_is_valid(obs, _BUILD_BARRACKS):
                
                unit_y, unit_x = get_units(obs, _TERRAN_COMMANDCENTER)
                
                offset_x = np.random.randint(-4, 4)
                offset_y = np.random.randint(-4, 4)
                
                target = (np.min(unit_x)+offset_x, np.min(unit_y)+offset_y)
        
                return actions.FunctionCall(_BUILD_BARRACKS, [_SCREEN, target])

        elif not at_max_supply(obs):
            if not self.barracks_selected:
                unit_y, unit_x = get_units(obs, _TERRAN_BARRACKS)
                
                if unit_y.any():
                    target = [int(unit_x.mean()), int(unit_y.mean())]
                    self.barracks_selected = True
                    return actions.FunctionCall(_SELECT_POINT, [_SCREEN, target])

            elif action_is_valid(obs, _TRAIN_MARINE):
                return actions.FunctionCall(_TRAIN_MARINE, [_QUEUED])

        elif at_max_supply(obs):
            if action_is_valid(obs, _ATTACK_MINIMAP):
                if self.base_top_left:
                    return actions.FunctionCall(_ATTACK_MINIMAP, [_MINIMAP, [39, 45]])
                return actions.FunctionCall(_ATTACK_MINIMAP, [_MINIMAP, [21, 24]])

            elif action_is_valid(obs, _SELECT_ARMY):
                self.barracks_selected = False
                return actions.FunctionCall(_SELECT_ARMY, [_NOADD])

        return actions.FunctionCall(_NO_OP, [])


def run_thread(agent_cls, map_name, visualize):
  with sc2_env.SC2Env(
      map_name=map_name,
      agent_race=RACE,
      bot_race=OPPONENT_RACE,
      difficulty=DIFFICULTY,
      step_mul=8,
      game_steps_per_episode=0,
      screen_size_px=(84, 84),
      minimap_size_px=(64, 64),
      visualize=visualize,
      camera_width_world_units=128) as env:
    env = available_actions_printer.AvailableActionsPrinter(env)
    agent = agent_cls()
    run_loop.run_loop([agent], env, 2500)
    if True:
      env.save_replay(agent_cls.__name__)


def main(unused_argv):
  """Run an agent."""
  stopwatch.sw.enabled = False
  stopwatch.sw.trace = False

  maps.get(MAP)  # Assert the map exists.

  agent_cls = HelloWorld

  threads = []
  for _ in range(1 - 1):
    t = threading.Thread(target=run_thread, args=(agent_cls, MAP, False))
    threads.append(t)
    t.start()

  run_thread(agent_cls, MAP, True)

  for t in threads:
    t.join()

  if False:
    print(stopwatch.sw)

if __name__ == "__main__":
  app.run(main)
