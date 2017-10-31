import threading
import time

import numpy as np

from pysc2 import maps
from pysc2.agents import base_agent
from pysc2.agents import random_agent
from pysc2.lib import actions
from pysc2.lib import features
from pysc2.env import available_actions_printer
from pysc2.env import run_loop
from pysc2.env import sc2_env
from pysc2.lib import stopwatch
from pysc2.env.environment import StepType

from absl import app

# Options
MAP = 'DefeatZerglingsAndBanelings'
RACE = 'T'
OPPONENT_RACE = 'Z'
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

# class DefeatZergling(base_agent.BaseAgent):
#     pass

def main(unused_argv):
    """Run an agent."""
    maps.get(MAP)

    with sc2_env.SC2Env(
        map_name=MAP,
        agent_race=RACE,
        bot_race=OPPONENT_RACE,
        difficulty=DIFFICULTY,
        step_mul=8,
        game_steps_per_episode=0,
        screen_size_px=(84, 84),
        minimap_size_px=(64, 64),
        visualize=True,
        camera_width_world_units=128) as env:
        env = available_actions_printer.AvailableActionsPrinter(env)
        agent = random_agent.RandomAgent()
        run_loop.run_loop([agent], env, 2500)

if __name__ == "__main__":
    app.run(main)
