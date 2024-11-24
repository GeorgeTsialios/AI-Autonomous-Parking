# This is going to be a parking game where the player has to park the car in the parking lot. The player car will be controlled by the arrow keys.
# The car's movement should resemble closely real-life physics in terms of acceleration, deceleration, and turning.
# There will be 10 parking spots, but only one free for the player, while the rest will be occupied by other cars. The free parking spot 
# will contain a rectangle, which will turn green once the player places the car inside it. The game resets once the car is stationary inside it for 1 second.
# The free parking spot will be randomly chosen at the beginning of each game. Also, the player car will spawn at a random position at the beginning of each game.
# There will be collision detection between the player car and the other cars, as well as the parking lot borders and a garden (depicted as a rectangle
# in the middle part of the parking lot). The car will have 8 depth sensors (radars), which will detect the distance to the nearest object (or window borders) in 8 directions.
# The radars will be drawn on the screen as lines.
# The player car will be controlled by an AI agent, which will use DDPG to learn how to park the car in the parking spot. The agent will have 9 actions: move forward, move backward, turn left, turn right, move forward and turn left, move forward and turn right, move backward and turn left, move backward and turn right, do nothing. 
# The agent's states will consist of the 8 depth sensors, the offset between the center of the car and the center of the parking spot in the x axis, the offset between the center of the car and the center of the parking spot in the y axis, the velocity of the car, the angle of the car. However, these features will be discretized into a smaller number of bins. This way we can reduce the state space size. 


import os
import pygame
import time
import math
import sys
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from enum import Enum
from stable_baselines3 import DDPG
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import torch as th


# Register this module as a gym environment. Once registered, the id is usable in gym.make().
register(
    id='parking-game-v0',                          
    entry_point='DDPG-instant:ParkingGameEnv', # module_name:class_name
)

pygame.mixer.init()

music = pygame.mixer.music.load("parking_game/sounds/1-Happy-walk.mp3")
# pygame.mixer.music.play(-1)     # -1 means that the music will loop indefinitely
pygame.mixer.music.set_volume(0.05)
collision_sound = pygame.mixer.Sound("parking_game/sounds/Car_Door_Close.wav")
start_up_sound = pygame.mixer.Sound("parking_game/sounds/carengine-5998-[AudioTrimmer.com].wav")
start_up_sound.set_volume(0.2)
green_sound = pygame.mixer.Sound("parking_game/sounds/success-bell-6776_8ODfLqon.wav")
green_sound.set_volume(0.1)


def scale_image(img, factor):
    size = round(img.get_width() * factor), round(img.get_height() * factor)        # size is a tuple of 2 integers, the new width and height of the image
    return pygame.transform.scale(img, size)  

pygame.font.init()      # Initialize the font module, essential for rendering text on the screen
SMALL_FONT = pygame.font.SysFont("gadugi", 16, False, False)    # Select text font, size, bold, italic.
LARGE_FONT = pygame.font.SysFont("gadugi", 22, False, False) 

PARKING_LOT = pygame.image.load("parking_game/imgs/stats/parking-lot-stats.png")
GARDEN_BORDER = pygame.image.load("parking_game/imgs/garden-border.png")
GARDEN_BORDER_MASK = pygame.mask.from_surface(GARDEN_BORDER)


GRAY_CAR = [scale_image(pygame.image.load("parking_game/imgs/car-gray-wheels.png"), 40/161), scale_image(pygame.image.load("parking_game/imgs/car-gray-wheels-right.png"), 40/161)]            # factor is equal to desired width of car / actual width of image
YELLOW_CAR = scale_image(pygame.image.load("parking_game/imgs/car-yellow-wheels.png"), 40/162)       # this way all cars have the same width (40px) 
PINK_CAR = scale_image(pygame.image.load("parking_game/imgs/car-pink-wheels.png"), 40/162)
GREEN_CAR = scale_image(pygame.image.load("parking_game/imgs/car-green-new-wheels.png"), 40/127)
PURPLE_CAR = scale_image(pygame.image.load("parking_game/imgs/car-purple-wheels.png"), 40/164)


GARDEN = pygame.Rect(125, 325, 500, 100)
TOP_RECT = pygame.Rect(0, -10, 750, 10)
BOTTOM_RECT = pygame.Rect(0, 750, 750, 10)
LEFT_RECT = pygame.Rect(-10, 0, 10, 750)
RIGHT_RECT = pygame.Rect(750, 0, 10, 750)


WIN_WIDTH, WIN_HEIGHT = PARKING_LOT.get_width(), PARKING_LOT.get_height()
WIN = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
pygame.display.set_caption("Parking Game!")

CAR_WIDTH, CAR_HEIGHT = 40, 81.24
cars = [YELLOW_CAR, PINK_CAR, GREEN_CAR, PURPLE_CAR,]     # flip() is used to flip the image vertically

free_spot_color = (255, 0, 0, 255)  
parking_spots = {}
intersection = None
free_spot_rect = None
FREE_SPOT_BORDER_MASK = None
PARKING_LOT_BORDER_MASK = None


class SkipEnv(gym.Wrapper[np.ndarray, int, np.ndarray, int]):
    """
    Return only every ``skip``-th frame (frameskipping)

    :param env: Environment to wrap
    :param skip: Number of ``skip``-th frame
        The same action will be taken ``skip`` times.
    """

    def __init__(self, env: gym.Env, skip: int = 4) -> None:
        super().__init__(env)
        assert env.observation_space.dtype is not None, "No dtype specified for the observation space"
        assert env.observation_space.shape is not None, "No shape defined for the observation space"
        self._skip = skip

    def step(self, action: int):
        """
        Step the environment with the given action
        Repeat action, sum reward, and observe the final state.
        
        :param action: the action
        :return: observation, reward, terminated, truncated, information
        """
        total_reward = 0.0
        for _ in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated:
                break

        return obs, total_reward, terminated, truncated, info


class ParkingGameEnv(gym.Env):
    # metadata is a required attribute
    # render_modes in our environment is either None or 'human'.
    # render_fps must be a non-zero value.
    metadata = {"render_modes": ["human"], 'render_fps': 20}

    def __init__(self, render_mode=None):

        self.render_mode = render_mode

        # Initialize the Parking problem
        self.car = AgentCar(6, fps=self.metadata['render_fps'])
        # self.car = PlayerCar(6, fps=self.metadata['render_fps'])            # change to this to control the car with arrow keys

        # Gym requires defining the action space. The action space is the agent's set of possible actions.
        # Training code can call action_space.sample() to randomly select an action. 
        self.action_space = spaces.Box(
            low = np.array([-1, -1]),
            high = np.array([1,  1]),
            shape = (2,),
            dtype = np.float16)

        # Gym requires defining the observation space. The observation space consists of the agent's set of possible positions.
        # The observation space is used to validate the observation returned by reset() and step().
        # Use a 1D vector: [radar0, radar1, radar2, radar3, radar4, radar5, radar6, radar7, offset_x, offset_y, velocity, angle]
        self.observation_space = spaces.Box(                                                # Box for continuous observation space
            low = np.array([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]),
            high = np.array([1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1]),
            shape = (12,),
            dtype = np.float16
        )

        self.clock = pygame.time.Clock()
        self.max_steps = 600
        self.successes = 0
        self.times_list = []            # times for each successful parking
        self.collisions_list = []       # number of collisions for each successful parking
        self.action = ""
        self.steps_value = 1
    
    def initialize_game(car_spawn_index):
        start_up_sound.play()

        random.shuffle(cars)
        global parking_spots
        global free_spot_rect
        global FREE_SPOT_BORDER_MASK
        global PARKING_LOT_BORDER_MASK

        parking_spots = {1: [pygame.Rect(158.33, 210.89, CAR_WIDTH, CAR_HEIGHT), pygame.transform.flip(cars[0], False, random.choice([True,False])), 158.33, 210.89],
                        2: [pygame.Rect(256.66, 210.89, CAR_WIDTH, CAR_HEIGHT), pygame.transform.flip(random.choice(cars), False, random.choice([True,False])),  256.66, 210.89],
                        3: [pygame.Rect(354.99, 210.89, CAR_WIDTH, CAR_HEIGHT), pygame.transform.flip(random.choice(cars), False, random.choice([True,False])), 354.99, 210.89],
                        4: [pygame.Rect(453.32, 210.89, CAR_WIDTH, CAR_HEIGHT), pygame.transform.flip(cars[1], False, random.choice([True,False])), 453.32, 210.89],
                        5: [pygame.Rect(551.65, 210.89, CAR_WIDTH, CAR_HEIGHT), pygame.transform.flip(random.choice(cars), False, random.choice([True,False])), 551.65, 210.89],
                        6: [pygame.Rect(158.33, 457.88, CAR_WIDTH, CAR_HEIGHT), pygame.transform.flip(random.choice(cars), False, random.choice([True,False])), 158.33, 457.88],
                        7: [pygame.Rect(256.66, 457.88, CAR_WIDTH, CAR_HEIGHT), pygame.transform.flip(cars[2], False, random.choice([True,False])), 256.66, 457.88],
                        8: [pygame.Rect(354.99, 457.88, CAR_WIDTH, CAR_HEIGHT), pygame.transform.flip(random.choice(cars), False, random.choice([True,False])), 354.99, 457.88],
                        9: [pygame.Rect(453.32, 457.88, CAR_WIDTH, CAR_HEIGHT), pygame.transform.flip(random.choice(cars), False, random.choice([True,False])), 453.32, 457.88],
                        10: [pygame.Rect(551.65, 457.88, CAR_WIDTH, CAR_HEIGHT), pygame.transform.flip(cars[3], False, random.choice([True,False])), 551.65, 457.88]}

        free_spot_index = random.randint(1, 10)   # the free spot will be randomly chosen at the beginning of each game
        # print(f"Free spot: {free_spot_index}")
        parking_spots.pop(free_spot_index)

        SPOT_RECTANGLES = [pygame.Rect(140.83, 201.5, 75, 100), pygame.Rect(239.16, 201.5, 75, 100), pygame.Rect(337.49, 201.5, 75, 100), pygame.Rect(435.82, 201.5, 75, 100), pygame.Rect(534.15, 201.5, 75, 100), pygame.Rect(140.83, 448.5, 75, 100), pygame.Rect(239.16, 448.5, 75, 100), pygame.Rect(337.49, 448.5, 75, 100), pygame.Rect(435.82, 448.5, 75, 100), pygame.Rect(534.15, 448.5, 75, 100)]

        free_spot_rect = SPOT_RECTANGLES[free_spot_index - 1]     # the rectangle that will turn green when the player parks the car inside it
        FREE_SPOT_BORDER = pygame.image.load(f"parking_game/imgs/free-spot-border-{free_spot_index}.png")       
        FREE_SPOT_BORDER_MASK = pygame.mask.from_surface(FREE_SPOT_BORDER)              

        PARKING_LOT_BORDER = pygame.image.load(f"parking_game/imgs/parking-lot-border-{free_spot_index}.png")
        PARKING_LOT_BORDER_MASK = pygame.mask.from_surface(PARKING_LOT_BORDER)
        return free_spot_index

    # Gym required function (and parameters) to reset the environment
    def reset(self, seed=None, options=None):
        super().reset(seed=seed) # gym requires this call to control randomness and reproduce scenarios.

        # Reset the Parking Optionally, pass in seed control randomness and reproduce scenarios.
        self.car.reset(seed=seed)

        self.current_step = 0

        # Construct the observation state:
       # [radar0, radar1, radar2, radar3, radar4, radar5, radar6, radar7, offset_x, offset_y, velocity, angle]
        self.state = list(self.car.discretize_state())
        obs = np.array(self.state).astype(np.float16)

        self.reward = 0
        self.collisions = 0
        
        # Additional info to return. For debugging or whatever.
        info = {}

        # Return observation and info
        return obs, info
    
    # Gym required function (and parameters) to perform an action
    def step(self, action):

        # Additional info to return. For debugging
        info = {}

        self.action = ""
        self.action = self.name_action(action)
        # Render environment
        if(self.render_mode == 'human'):
            self.render()                   # Render the environment using previous state, reward but current action. This way we show the aprropriate input-output of the NN.

        for event in pygame.event.get():            
            if event.type == pygame.QUIT:       # If the user closes the window, the game stops
                pygame.quit()
                sys.exit()

        self.current_step += 1
        self.car.check_radars(PARKING_LOT_BORDER_MASK)
        self.car.move_player(action)                            
        # self.car.move_player()                            # Change to this to control the car with arrow keys
        terminated, collides, inside_spot = self.car.check_collision()

        # Construct the observation state:
        # [radar0, radar1, radar2, radar3, radar4, radar5, radar6, radar7, offset_x, offset_y, velocity, angle]
        self.state = list(self.car.discretize_state())
        obs = np.array(self.state).astype(np.float16)
    
        # Calculate reward
        self.reward = 0

        if terminated:
            # print("TERMINATED", end=" ")
            info["is_success"] = True
            self.successes += 1
            self.times_list.append(self.current_step/20)
            self.collisions_list.append(self.collisions)
            self.reward += 5000
            if self.state[10] < 0.12:
                # print("REVERSE", end=" ")
                self.reward += 100
            if self.render_mode == 'human':
                self.render(terminated = True)

        # elif inside_spot:
        #     self.reward += 2 + 2 / (abs(self.state[10] - 0.4) + 1)                      # reward for being inside the parking spot
        #     if self.state[10] == 0.4:                   # extra reward for being stationary
        #         # print("BEING STATIONARY INSIDE PARKING SPOT", end=" ")
        #         self.reward += 5

        else:
            # reward -= (self.car.distance / 730.26) * 5        # punishment for being away from the center of the parking spot (730.26 is the max distance)
            self.reward -= abs(self.state[8] - 0.5) * 6
            self.reward -= abs(self.state[9] - 0.5) * 6

            if self.state[10] > 0.5:
                self.reward += 2

            if collides:
                self.collisions += 1
                # print("Collision", end=" ")
                self.reward -= 500             # punish the car for colliding with an object

            if abs(self.state[8] - 0.5) >= 0.1 or abs(self.state[9] - 0.5) >= 0.1:    # when the car is far away from the parking spot
                if self.state[10] > 0.25 and self.state[10] < 0.55:    # punish the car for moving too slow
                     # print("BEING STATIONARY FAR AWAY PARKING SPOT", end=" ")
                    self.reward -= 5
                    if self.state[10] == 0.4:
                        self.reward -= 5
            else:                     # when the car is near the parking spot
                if self.state[10] > 0.35 and self.state[10] < 0.45:    # punish the car for moving too slow
                     # print("BEING STATIONARY NEAR PARKING SPOT", end=" ")
                    self.reward -= 3
                    if self.state[10] == 0.4:
                        self.reward -= 5
                if (self.state[11] > 0.98 or self.state[11] < 0.02) or (self.state[11] > 0.48 and self.state[11] < 0.52):      # reward the car for being in the right angle
                    self.reward += 0.5
        
        if self.current_step >= self.max_steps:
            if self.render_mode == 'human':
                self.render(truncated = True)
            info["is_success"] = False
            truncated = True
            terminated = True
            info['truncated'] = True
        else:
            truncated = False
            info['truncated'] = False

        # Return observation, reward, terminated, truncated, info
        return obs, self.reward, terminated, truncated, info
    
    # Gym required function to render environment
    def render(self, terminated=False, truncated=False):
        global free_spot_color
        WIN.blit(PARKING_LOT, (0, 0))

        for index, spot in parking_spots.items():
            WIN.blit(spot[1], (spot[2], spot[3]))
        
        if not terminated:
            free_spot_color = (255, 0, 0)
        pygame.draw.rect(WIN, free_spot_color, free_spot_rect, 2)
            
        self.car.draw()
        # pygame.draw.circle(WIN, (0, 0, 255), free_spot_rect.center, 3)       # draw the center of the parking spot with blue color
        if intersection is not None:
            pygame.draw.rect(WIN, (255, 0, 0), intersection)            # draw a red rectangle around the point of intersection

        Algorithm_text = LARGE_FONT.render(f"Algorithm", 1, "white")   # 1 is the anti-aliasing level, keep it 1 for better quality
        WIN.blit(Algorithm_text, (780, 12))          
        Semicolon_large_text = LARGE_FONT.render(f":", 1, "white")   
        WIN.blit(Semicolon_large_text, (920, 12))    
        DDPG_text = LARGE_FONT.render(f"DDPG", 1, "white") 
        WIN.blit(DDPG_text, (939, 12))

        Steps_trained_text = LARGE_FONT.render(f"Steps trained", 1, "white")   # 1 is the anti-aliasing level, keep it 1 for better quality
        WIN.blit(Steps_trained_text, (780, 43))          
        WIN.blit(Semicolon_large_text, (920, 43))    
        Steps_valuetext = LARGE_FONT.render(f"{self.steps_value}", 1, "white") 
        WIN.blit(Steps_valuetext, (939, 43))  

        pygame.draw.line(WIN, "white", (750, 87), (1100, 87), 1)    

        Step_text = LARGE_FONT.render(f"Step", 1, "white")   
        WIN.blit(Step_text, (780, 102))
        WIN.blit(Semicolon_large_text, (830, 102))  
        if not terminated:
            step_to_print = self.current_step - self.current_step % 4               # print the step every 4 frames because of the frameskip = 4
        else:
            step_to_print = self.current_step
        Current_step_text = LARGE_FONT.render(f"{step_to_print}", 1, "white")   
        WIN.blit(Current_step_text, (849, 102))

        Time_text = LARGE_FONT.render(f"Time", 1, "white")   
        WIN.blit(Time_text, (928, 102))
        WIN.blit(Semicolon_large_text, (981, 102))    
        Time_value_text = LARGE_FONT.render(f"{self.current_step/20:.2f}s", 1, "white")   
        WIN.blit(Time_value_text, (1000, 102))        

        Inputs_text = LARGE_FONT.render(f"Inputs", 1, "white")
        WIN.blit(Inputs_text, (890, 146))
        Semicolon_small_text = SMALL_FONT.render(f":", 1, "white")
        Inputs_names = ["Radar 1", "Radar 2", "Radar 3", "Radar 4", "Radar 5", "Radar 6", "Radar 7", "Radar 8", "Offset_x", "Offset_y", "Velocity", "Angle"]

        if self.current_step % 4 == 0 or terminated:       # save the state and reward every 4 frames because of the frameskip = 4
            self.state_saved = self.state
            self.reward_saved = self.reward

        for i in range(12):
            Input_text = SMALL_FONT.render(f"{Inputs_names[i]}", 1, "white")
            WIN.blit(Input_text, (780, 185 + 25 * i))
            WIN.blit(Semicolon_small_text, (846, 185 + 25 * i))
            Inputs_value_text = SMALL_FONT.render(f"{self.state_saved[i]:.3f}", 1, "white")
            WIN.blit(Inputs_value_text, (863, 185 + 25 * i))

        Rewards_text = LARGE_FONT.render(f"Rewards", 1, "white")
        WIN.blit(Rewards_text, (886, 503))
        Reward_text = SMALL_FONT.render(f"Reward", 1, "white")
        WIN.blit(Reward_text, (780, 535))
        WIN.blit(Semicolon_small_text, (846, 535))
        Reward_value_text = SMALL_FONT.render(f"{self.reward_saved:.2f}", 1, "white")
        WIN.blit(Reward_value_text, (863, 535))

        Outputs_text = LARGE_FONT.render(f"Outputs", 1, "white")
        WIN.blit(Outputs_text, (884, 581))

        square_coordinates = [(900, 623), (840, 683), (900, 683), (960, 683)]
        square_colors = ["black", "black", "black", "black"] if not (terminated or truncated) else ["gray", "gray", "gray", "gray"]
        triangle_coordinates = [[(925, 641), (917, 655), (933, 655)], [(858,708), (872, 716), (872, 700)], [(916,701), (933, 701), (925, 715)], [(978,716), (978, 700), (992, 708)]]

        if not (terminated or truncated):
            if "UP" in self.action:
                square_colors[0] = "green"
            elif "DOWN" in self.action:
                square_colors[2] = "green"
            if "LEFT" in self.action:
                square_colors[1] = "green"
            elif "RIGHT" in self.action:
                square_colors[3] = "green"

        for i in range(4):
            pygame.draw.rect(WIN, square_colors[i], (square_coordinates[i][0], square_coordinates[i][1], 50, 50), 0)
            pygame.draw.rect(WIN, "white", (square_coordinates[i][0], square_coordinates[i][1], 50, 50), 1)
            pygame.draw.polygon(WIN, "white", triangle_coordinates[i], 0)

        pygame.display.update()

        if terminated or truncated:
            # blur the game screen
            surf = pygame.Surface((WIN_WIDTH, WIN_HEIGHT))
            surf.set_alpha(200)
            surf.fill((0, 0, 0))
            WIN.blit(surf, (0, 0))
            if terminated:          # draw a large green tick on the center of the screen
                pygame.draw.line(WIN, "green", (750 // 2 - 100, WIN_HEIGHT // 2), (750 // 2 - 40, WIN_HEIGHT // 2 + 100), 5)
                pygame.draw.line(WIN, "green", (750 // 2 - 40, WIN_HEIGHT // 2 + 100), (750 // 2 + 100, WIN_HEIGHT // 2 - 100), 5)
            else:                   # draw a large red cross on the center of the screen
                pygame.draw.line(WIN, "red", (750 // 2 - 100, WIN_HEIGHT // 2 - 100), (750 // 2 + 100, WIN_HEIGHT // 2 + 100), 5)
                pygame.draw.line(WIN, "red", (750 // 2 - 100, WIN_HEIGHT // 2 + 100), (750 // 2 + 100, WIN_HEIGHT // 2 - 100), 5)  
            pygame.display.update()
            pygame.time.wait(2000)     # freeze the screen for 2s

        self.clock.tick(self.car.fps)
    
    def name_action(self, action):         # transform the action from continuous to discrete
        if action[1] >= -0.33 and action[1] <= 0.33 and action[0] >= -0.33 and action[0] <= 0.33:
            self.action = "NOTHING"
        else:
            if action[1] > 0.33:
                self.action = "UP"
            elif action[1] < -0.33:
                self.action = "DOWN"
            
            if action[0] < -0.33:
                self.action += " LEFT"
            elif action[0] > 0.33:
                self.action += " RIGHT"
            
            self.action.strip()    # remove the leading and trailing whitespaces

        return self.action
            
        


class AbstractCar:
    def __init__(self, max_vel, fps):
        self.img = self.IMG
        self.max_vel = max_vel
        self.vel = 0
        self.acceleration = 0.1
        self.fps = fps
        self.distance = 0
        self.count = 1
        self.difference = None
        self.t_start = None
        global new_img
        new_img = None

    def calculate_START_POS(self):       
        SPAWN_RECTS = [pygame.Rect(2, 2, 651, 95), pygame.Rect(2, 558, 651, 95), pygame.Rect(2, 97, 26, 461), pygame.Rect(627, 97, 26, 461)]        # these are the rectangles where the car can spawn
        car_spawn_index = random.randint(0, 3)
        car_spawn = SPAWN_RECTS[car_spawn_index]
        car_spawn.x += random.randint(0, car_spawn.width)               # randomize the spawn position of the player car
        car_spawn.y += random.randint(0, car_spawn.height)
        # print(f"Car spawn: {car_spawn.x}, {car_spawn.y}")
        rotated_image = pygame.transform.rotate(self.img, self.angle)
        new_rect = rotated_image.get_rect(center=self.img.get_rect(topleft= (car_spawn.x, car_spawn.y)).center)
        new_rect.topleft = (car_spawn.x, car_spawn.y)
        # print(f"New_rect x: {new_rect.x}, y: {new_rect.y}")

        return new_rect.x + new_rect.width / 2 - self.img.get_width() / 2, new_rect.y + new_rect.height / 2 - self.img.get_height() / 2, car_spawn_index

    def draw(self):
        global new_img
        # pygame.draw.rect(WIN, (0, 0, 0), new_img[1])                    # draw the new_rect rectangle around the car
        # pygame.draw.circle(WIN, (255, 0, 0), new_img[1].topleft, 5)     # draw the new_rect.x and new_rect.y coordinates with red color
        # pygame.draw.circle(WIN, (0, 0, 255), (self.x, self.y), 5)       # draw the self.x and self.y coordinates with blue color
        if new_img is not None:
            WIN.blit(new_img[0], new_img[1].topleft)
            # pygame.draw.circle(WIN, (0, 255, 0), self.center, 3)       # draw the center of the car with green color
            for radar in self.radars:
                    position = radar[0]
                    pygame.draw.line(WIN, (0, 255, 0), self.center, position, 1)
                    pygame.draw.circle(WIN, (0, 255, 0), position, 3)
    
    def rotate_center(self):
        '''
        This function rotates an image around its center and blits it to the window.
        '''        
        rotated_image = pygame.transform.rotate(self.img, self.angle)
        new_rect = rotated_image.get_rect(center=self.img.get_rect(topleft=(self.x, self.y)).center)
        new_mask = pygame.mask.from_surface(rotated_image)      # create a new mask for the rotated image. This is necessary for pixel perfect collision detection.
        self.center = (new_rect.x + new_rect.width // 2, new_rect.y + new_rect.height // 2)
        # print(f"New_rect x: {new_rect.x}, y: {new_rect.y}")
        # print(f"Car spawn: {car_spawn.x}, {car_spawn.y}")
        return rotated_image, new_rect, new_mask
   
    def check_collision(self):
        global new_img
        global intersection

        new_img = self.rotate_center()
        collides = False
        inside_spot = False
        terminated = False

        if self.collide_map(new_img[1], new_img[2]):
            collision_sound.set_volume(max(min(abs(self.vel * 0.01), 0.02), 0.008))
            collision_sound.play()
            collides = True
            self.bounce()
        
        elif new_img[1].x < 0 or new_img[1].x > WIN_WIDTH or new_img[1].y < 0 or new_img[1].y > WIN_HEIGHT:       # if the car goes out of the window, return_to_map it
            print(f"out of bounds - x: {new_img[1].x}, y: {new_img[1].y}")
            intersection =  None
            self.return_to_map()

        else:
             self.last_x, self.last_y = self.x, self.y      # save the last, safe position of the car (where it does not collide with anything)
             intersection = None

        global free_spot_color

        if self.collide_free_spot(new_img[1], new_img[2]):
            # if free_spot_color == (255, 0, 0):         # if the color is red, it means that the car has just parked in the spot, so play the sound
            green_sound.play()
            free_spot_color = (0, 255, 0)
            inside_spot = True                          
            terminated = True
            return terminated, collides, inside_spot
        else:
            free_spot_color = (255, 0, 0)
            self.count = 1

        return terminated, collides, inside_spot
    
    def collide_map(self, new_rect, new_mask):
        global intersection
        offset = (int(new_rect.x), int(new_rect.y))     # offset is the difference between the top left corner of the car image and the top left corner of the window (border mask)
        
        if new_rect.colliderect(GARDEN):                # colliderect() is less computanionally expensive than pixel perfect collision detection using masks. That's why we first check if the rectangles collide.
            # pygame.draw.rect(WIN, (0, 0, 0), new_rect)
            if GARDEN_BORDER_MASK.overlap(new_mask, offset) is not None:   # now we check for pixel perfect collision, because when the car is turning, the new_rect rectangle is bigger than the car image. This leads to false positive collision detetctions when the car is turning around the edges of the garden.
                intersection = new_rect.clip(GARDEN)                       # returns a new rectangle that represents the intersection of the two rectangles.
                # print(f"collision with garden")
                return True
        elif new_rect.colliderect(TOP_RECT):
                intersection = None
                # print(f"collision with top rect")
                # print(f"x: {new_rect.x}, y: {new_rect.y}")
                return True
        elif new_rect.colliderect(BOTTOM_RECT):
                intersection = None
                # print(f"collision with bottom rect")
                # print(f"x: {new_rect.x}, y: {new_rect.y}")
                return True
        elif new_rect.colliderect(LEFT_RECT):
                intersection = None
                # print(f"collision with left rect")
                # print(f"x: {new_rect.x}, y: {new_rect.y}")
                return True
        elif new_rect.colliderect(RIGHT_RECT):
                intersection = None
                # print(f"collision with right rect")
                # print(f"x: {new_rect.x}, y: {new_rect.y}")
                return True
        for index, spot in parking_spots.items():
            if new_rect.colliderect(spot[0]):
                if PARKING_LOT_BORDER_MASK.overlap(new_mask, offset) is not None: 
                    intersection = new_rect.clip(spot[0])
                    # print(f"collision with car {index}")
                    # print(f"x: {new_rect.x}, y: {new_rect.y}")
                    return True
        return False

    def collide_free_spot(self, new_rect, new_mask):
        offset = (int(new_rect.x), int(new_rect.y))
        if new_rect.colliderect(free_spot_rect):
            if FREE_SPOT_BORDER_MASK.overlap(new_mask, offset) is None:
                return True
        return False

    def bounce(self):
        '''
        This method tries to move the car away from the object it is colliding with.
        To do so, it changes the velocity of the car, so that it moves in the opposite direction.
        The car will keep moving in this direction until it is no longer colliding with the object.
        After that, the car will stop moving.
        '''
        # print(f"{self.vel:.2f}")
        self.vel = -self.vel                            # reverse the direction of the car, so that it exits from colliding 
        if round(self.vel, 2) == 0.00:         # this was used for when the car was stuck colliding while having velocity = 0, the game would crash
            self.vel = -0.1       # however I think this is not necessary anymore, because the car will always have a velocity different from 0 (you can not press the up arrow key and the down arrow key at the same time)
        counter = 0
        while True:
            # print(f"{self.vel:.2f}")
            counter += 1
            if counter == 50:                           # if the car is stuck in an infinite loop, break it. This happens when the car was colliding with the object while moving away from it. For example, the car would be moving in reverse and turning at the same time. Its rotation eould make it so that its front car would be colliding with the object, while its back would be moving away from it. So the switching in its velocity in line 80 woul be a mistake and would force the car to move into the object. That's why, if the while loop runs for too long, we assume that this is the issue and we switch the velocity again. 
                self.vel = -self.vel
            self.move()
            new_img = self.rotate_center()
            if not self.collide_map(new_img[1], new_img[2]) and (self.x, self.y) != (self.last_x, self.last_y):     # the 2nd condition is necessary, because the car would be stuck in an infinite loop when it was colliding with the object in 2 adjacent points, around one safe point. This would happen when the car was turning around the edges of the parking spots. It would collide with the parking spot in one point, then return to the safe point and when it moved in the opposite direction, it would collide in the 2nd point and return to the safe point again. This would repeat indefinitely.
                break
        self.vel = 0

    def rotate(self, left=False, right=False):
        if abs(self.vel) > 0:        # if the car is moving, it can rotate.
            turning_factor = 0.85 if self.vel <= self.max_vel * 2/3 else 0.5     # the turning factor depends on the velocity of the car. The higher the velocity, the less the car will turn. This is because the car has inertia and it is harder to turn it when it is moving faster.  
            if left:
                self.angle += turning_factor * self.vel              # When turning, the angle of the car changes depending on its velocity. The higher the velocity, the more the angle changes.
            elif right:
                self.angle -= turning_factor * self.vel

    def move_forward(self):
        self.vel = min(self.vel + self.acceleration, self.max_vel)
        self.move()

    def move_backward(self):    # move backward with half the max speed
        self.vel = max(self.vel - self.acceleration, -self.max_vel * 2/3)   # max because these are negative values, we are still choosing the lower speed
        self.move()

    def move(self):
        angle_radians = math.radians(self.angle)
        vertical_distance = math.cos(angle_radians) * self.vel       # the distance to move in the y direction
        horizontal_distance = math.sin(angle_radians) * self.vel     # the distance to move in the x direction

        self.y -= vertical_distance
        self.x -= horizontal_distance
    
    def reduce_speed(self):
        if self.vel > 0:
            self.vel = max(self.vel - self.acceleration / 2, 0)     # friction is half the acceleration
        else:
            self.vel = min(self.vel + self.acceleration / 2, 0)
        self.move()

    def return_to_map(self):
        self.img = self.IMG
        self.vel = 0
        self.angle = random.randint(0, 360)
        self.x, self.y, car_spawn_index = self.calculate_START_POS()
        self.last_x, self.last_y = self.x, self.y
        self.rotate_center()
        return car_spawn_index

    def reset(self, seed = None):
        random.seed(seed)
        car_spawn_index = self.return_to_map()
        ParkingGameEnv.initialize_game(car_spawn_index)
        self.check_radars(PARKING_LOT_BORDER_MASK)
        global new_img
        new_img = self.rotate_center()

    def check_radars(self, game_map):
        self.radars = [[(0, 0), 0] for _ in range(8)]
        degrees = [45, 75, 105, 135, 225, 255, 285, 315]
        step_size = 20
        for i in range(8):
            length = 30 if degrees[i] % 45 == 0 else 45                 # the starting length of the radar depends on its angle (because we don't want the radar to start checking inside the car)
            offset = 25 if degrees[i] % 45 == 0 else 40                 # offset is the distance from the center of the car to the edge of the car image. It depends on the angle of the radar. We use it to calculate the real distance of the radar.
            collide = False

            while True:
                test_x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degrees[i]))) * length)
                test_y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degrees[i]))) * length)
                if (test_x <= 0 or test_x >= 750) or (test_y <= 0 or test_y >= 750) or game_map.get_at((test_x, test_y)) != 0:
                    collide = True
                    break
                x = test_x
                y = test_y
                if length + step_size > 245:        # so that we achieve a max distance of 205
                    break
                length = length + step_size
            
            if collide:                             # if the radar collides with an object, we move it back by 1 repeatedly, until it is no longer colliding
                while length > 0:
                    length = length - 1
                    x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degrees[i]))) * length)
                    y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degrees[i]))) * length)
                    if (x > 0 and x < 750) and (y > 0 and y < 750) and game_map.get_at((x, y)) == 0:
                        break

            # dist = round(math.sqrt(math.pow(x - self.center[0], 2) + math.pow(y - self.center[1], 2)) - offset, 1)
            distance = length - offset          # the real distance of each radar is the length of the radar minus its offset
            # print(f" Radar {degrees[i]}: {distance}")
            self.radars[i] = [(x, y), distance]

    def discretize_state(self):
        # previous_distance = self.distance 
        for radar in self.radars:
            radar[1] = round(radar[1] / 205, 3)
        # print(f"Radar 2: {self.radars[1][1]}", end=" ") 
        discrete_vel = round((self.vel + 4) / 10, 3)     
        # print(f"Discrete_vel: {discrete_vel}")       
        discrete_angle = round( (((self.angle) % 360) / 360), 3)
        # print(f"Discrete_angle: {discrete_angle}", end=" ") 
        # self.distance = round( (math.sqrt(math.pow(self.center[0] - free_spot_rect.centerx, 2) + math.pow(self.center[1] - free_spot_rect.centery, 2)) / 730.26) , 2)    # the distance of the car to the center of the parking spot
        # print(f"Distance: {self.distance/ 730.26 :.2f}")
        # distance_discrete = self.distance // 100 + 9 if self.distance >= 100 else self.distance // 10          # The discretized distance has 17 bins, in range [0, 16]
        # print(f"Previous Distance {previous_distance}     Distance: {self.distance}     Self.vel {self.vel}")
        # self.difference = 1 if previous_distance - self.distance > 0  else -1 if previous_distance - self.distance < 0 else 0    # the difference between the previous distance and the current distance has 3 bins, in range [-1, 1]
        # print(f"Difference: {self.difference}", end=" ")
        offset_x = round(-((free_spot_rect.centerx - self.center[0]) - 551.65) / (551.65*2), 3)     # the offset of the car in the x direction has 2 bins, 0 if the car is to the left of the parking spot, 1 if the car is to the right of the parking spot
        offset_y = round(-((free_spot_rect.centery - self.center[1]) -478.5) / (478.5*2), 3)    # the offset of the car in the y direction has 2 bins, 0 if the car is above the parking spot, 1 if the car is below the parking spot    
        # print(f"Offset x: {offset_x} Offset y: {offset_y}")
        
        return self.radars[0][1], self.radars[1][1], self.radars[2][1], self.radars[3][1], self.radars[4][1], self.radars[5][1], self.radars[6][1], self.radars[7][1], offset_x, offset_y, discrete_vel, discrete_angle


class PlayerCar(AbstractCar):           # the player car will have additional methods for moving using the arrow keys
    IMG = GRAY_CAR[0]

    def move_player(self):
        self.player_action = ""
        keys = pygame.key.get_pressed()
        throttling = False   
        self.img = GRAY_CAR[0]           # the car image is set to the default image, so that it does not rotate when the player is not pressing the left or right arrow key     
        if not (keys[pygame.K_LEFT] and keys[pygame.K_RIGHT]): 
            if keys[pygame.K_LEFT]:                 # Keyboard ghosting is a hardware issue where certain combinations of keys cannot be detected simultaneously due to the design of the keyboard.
                    self.rotate(left=True)          # Due to this limitation of keyboard, we can only detect two arrow key presses at a time. This means that if the player is pressing the left and the right arrow key
                    self.img = pygame.transform.flip(GRAY_CAR[1], True, False)     # and then presses the up arrow key, the car will not move, as this third key press will not be detected.                                
                    self.player_action = "LEFT "
            elif keys[pygame.K_RIGHT]:                
                    self.rotate(right=True)
                    self.img = GRAY_CAR[1]   
                    self.player_action = "RIGHT "                             # we change the car img to the one that the wheels are turning
        if not (keys[pygame.K_UP] and keys[pygame.K_DOWN]):              # if both keys are pressed, the car should not move
            if keys[pygame.K_UP]:
                throttling = True
                self.move_forward()
                self.player_action = "UP " + self.player_action
            elif keys[pygame.K_DOWN]:
                throttling = True
                self.move_backward()
                self.player_action = "DOWN " + self.player_action

        if self.player_action == "":
            self.player_action = "NOTHING "
        if not throttling and self.vel !=0:                # if the player is not stepping on the gas, reduce the speed 
            self.reduce_speed()


class AgentCar(AbstractCar):
    IMG = GRAY_CAR[0]
    
    def move_player(self, agent_action):
        throttling = False   
        self.img = GRAY_CAR[0]           # the car image is set to the default image, so that it does not rotate when the player is not pressing the left or right arrow key     
        if agent_action[0] < - 0.333:                 # Move Left
                self.rotate(left=True)          
                self.img = pygame.transform.flip(GRAY_CAR[1], True, False)    
        elif agent_action[0] > 0.333:                  # Move Right 
                self.rotate(right=True)
                self.img = GRAY_CAR[1]                                # we change the car img to the one that the wheels are turning
        if agent_action[1] > 0.333:                    # Move Up
            throttling = True
            self.move_forward() 
        elif agent_action[1] < -0.333:                 # Move Down
            throttling = True
            self.move_backward()

        if not throttling and self.vel != 0:                # if the player is not stepping on the gas, reduce the speed 
            self.reduce_speed()



# Number of steps per episode taken by the agent to park.
# Since we operate at 20 fps, the agent chooses 20 actions per second. The car
# can always be parked in less than 30 seconds, so we will allow max 20 x 30 = 600 steps
max_steps = 600

models_dir = "parking_game/DDPG-models"
log_dir = "parking_game/DDPG-logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Train using DDPG algorithm (either from scratch or continue training)
def train_DDPG(steps_to_train, render=False, steps_previously_trained=0, run=1):

    env = gym.make('parking-game-v0', render_mode='human' if render else None)
    env = SkipEnv(env, skip=4)  # Skip 4 frames per step to speed-up training
    env = Monitor(env, info_keywords=("is_success",))  # Wrap the environment to log episode statistics
    # env = DummyVecEnv([lambda: env])  

    if steps_previously_trained > 0:
        model_path = f"{models_dir}/ddpg_model-{run}_{steps_previously_trained}_steps.zip"
        model = DDPG.load(model_path, env=env, tensorboard_log=log_dir, device="auto")  # Load the model
    else:
        # policy_kwargs = dict(activation_fn=th.nn.Tanh, net_arch=[128, 128, 128])       # change the policy network architecture to a 3-layer neural network with 128 units each
        model = DDPG("MlpPolicy", env, verbose=1, tensorboard_log=log_dir, device="auto")       # Create DDPG model, MlpPolicy is a neural network with 2 hidden layers of 64 units each, it is chosen because our input is a vector of 8 values and not an image
                                                                                              # Change device to "cuda" for GPU training or "cpu" for CPU training
    # print(model.policy) # print the model's network architecture

    checkpoint_callback = CheckpointCallback(save_freq=50000, save_path=models_dir, name_prefix=f'ddpg_model-{run}')

    model.learn(total_timesteps=steps_to_train, callback=checkpoint_callback, tb_log_name="DDPG", reset_num_timesteps=True)  # Train the model
    model.save(f"{models_dir}/ddpg_model-{run}_{steps_to_train}_steps.zip")  # Save the model
 
    print(f"Success / episodes: {env.unwrapped.successes} / {steps_to_train / max_steps :.0f}")


    env.close()



def test_DDPG(test_episodes, run=1, steps_trained=0, render=True):
    
    env = gym.make('parking-game-v0', render_mode='human' if render else None)
    env = SkipEnv(env, skip=4)  # Skip 4 frames per step to speed-up training
    env = Monitor(env)  # Wrap the environment to log episode statistics
    env.unwrapped.steps_value = steps_trained

    rewards = []
    model_path = f"{models_dir}/ddpg_model-{run}_{steps_trained}_steps.zip"
    model = DDPG.load(model_path, env=env, device="cuda")  

    for episode in range(67, test_episodes+1):
        terminated = False
        truncated = False
        total_reward = 0
        reward = 0
        state = env.reset(seed=episode)[0]

        while not terminated and not truncated:
            action,_ = model.predict(state, deterministic=True)
            # print(f"Step: {env.unwrapped.current_step:3d} State: {state} Reward: {reward:.2f} -> Action: {env.unwrapped.action:<10}")
            state, reward, terminated, truncated,_ = env.step(action)
            total_reward += reward
        
        rewards.append(total_reward)
        # if terminated:
        #     print(f"Episode {episode} Reward: {total_reward:.2f} Success")

    print(f"\nAlgorithm: DDPG\nSteps trained: {steps_trained}")
    print(f"Success ratio: {env.unwrapped.successes} / {test_episodes}")
    # print(f"Times list: {env.unwrapped.times_list}")
    print(f"Average successful episode time: {np.mean(env.unwrapped.times_list):.3f}")
    # print(f"Collisions list: {env.unwrapped.collisions_list}")
    print(f"Average successful episode collisions: {np.mean(env.unwrapped.collisions_list):.3f}")

def test_random_agent(test_episodes, render=True):
    env = gym.make('parking-game-v0', render_mode='human' if render else None)
    rewards = []

    for episode in range(2, test_episodes+1):
        terminated = False
        total_reward = 0
        state = env.reset(seed=episode)[0]

        while not terminated:
            random_action = env.action_space.sample()
            state,reward,terminated,_,_ = env.step(random_action)
            total_reward += reward
            # print(f"Step: {env.unwrapped.current_step} Action: {AgentAction(random_action).name:<10} -> State: {state}", end=' ')
            # print(f"State: {state}", end=' ')
            print(f'Reward: {reward:.2f}') 

        rewards.append(total_reward)
        if terminated:
            print(f"Episode {episode} Reward: {total_reward:.2f} Success")

    print("\n")
    for episode in range(1, test_episodes+1):
        print(f'Episode {episode} Reward: {rewards[episode-1]:.2f}')
    print(f"\nMean episode reward: {np.mean(rewards):.2f}")
    print(f"Success ratio: {env.unwrapped.successes} / {test_episodes}")


if __name__ == '__main__':

    # test_random_agent(10, render=True)
            
    # Train/test using DDPG
    # train_DDPG(7000000, render=False, steps_previously_trained=3550000, run=12)
    test_DDPG(100, run='4', steps_trained=1550000, render=True)