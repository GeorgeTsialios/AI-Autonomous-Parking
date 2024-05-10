# This is going to be a parking game where the player has to park the car in the parking lot. The player car will be controlled by the arrow keys.
# The car's movement should resemble closely real-life physics in terms of acceleration, deceleration, and turning.
# There will be 10 parking spots, but only one free for the player, while the rest will be occupied by other cars. The free parking spot 
# will contain a rectangle, which will turn green once the player places the car inside it. The game resets once the car is stationary inside it for 1 second.
# The free parking spot will be randomly chosen at the beginning of each game. Also, the player car will spawn at a random position at the beginning of each game.
# There will be collision detection between the player car and the other cars, as well as the parking lot borders and a garden (depicted as a rectangle
# in the middle part of the parking lot). The car will have 8 depth sensors (radars), which will detect the distance to the nearest object (or window borders) in 8 directions.
# The radars will be drawn on the screen as lines.
# The player car will be controlled by an AI agent, which will use Q-learning to learn how to park the car in the parking spot. The agent will have 9 actions: move forward, move backward, turn left, turn right, move forward and turn left, move forward and turn right, move backward and turn left, move backward and turn right, do nothing. 
# The agent's states will consist of the 8 depth sensors, the velocity of the car, the angle of the car and the distance between the center of the car and the center of the parking spot. However, these features will be discretized into a smaller number of bins. This way we can reduce the state space size. 
# The agent will have a Q-table, which will be updated after each action. The agent will have a reward system, which will give a reward of 100 if the car is parked in the parking spot, and -20 if the car collides with an object or goes out of the window. The agent will have a discount factor of 0.9 and a learning rate of 0.1. The agent will have an epsilon value of 0.1, which will be used for epsilon-greedy exploration. The agent will have a maximum of 400 episodes to learn how to park the car.

import pygame
import time
import math
import sys
import random
import pickle
import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import check_env
from enum import Enum

# Register this module as a gym environment. Once registered, the id is usable in gym.make().
register(
    id='parking-game-v0',                          
    entry_point='environment:ParkingGameEnv', # module_name:class_name
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


PARKING_LOT = pygame.image.load("parking_game/imgs/parking-lot.png")
GARDEN_BORDER = pygame.image.load("parking_game/imgs/garden-border.png")
GARDEN_BORDER_MASK = pygame.mask.from_surface(GARDEN_BORDER)


RED_CAR = [scale_image(pygame.image.load("parking_game/imgs/car-red-wheels.png"), 40/161), scale_image(pygame.image.load("parking_game/imgs/car-red-wheels-right.png"), 40/161)]            # factor is equal to desired width of car / actual width of image
YELLOW_CAR = scale_image(pygame.image.load("parking_game/imgs/car-yellow-wheels.png"), 40/162)       # this way all cars have the same width (40px) 
PINK_CAR = scale_image(pygame.image.load("parking_game/imgs/car-pink-wheels.png"), 40/162)
# GREEN_CAR = scale_image(pygame.image.load("parking_game/imgs/car-green.png"), 40/163)
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



class ParkingGameEnv(gym.Env):
    # metadata is a required attribute
    # render_modes in our environment is either None or 'human'.
    # render_fps must be a non-zero value.
    metadata = {"render_modes": ["human"], 'render_fps': 20}

    def __init__(self, render_mode=None):

        self.render_mode = render_mode

        # Initialize the Parking problem
        self.car = AgentCar(4, fps=self.metadata['render_fps'])
        # self.car = PlayerCar(6, fps=self.metadata['render_fps'])            # change to this to control the car with arrow keys

        # Gym requires defining the action space. The action space is the agent's set of possible actions.
        # Training code can call action_space.sample() to randomly select an action. 
        self.action_space = spaces.Discrete(len(AgentAction))

        # Gym requires defining the observation space. The observation space consists of the agent's set of possible positions.
        # The observation space is used to validate the observation returned by reset() and step().
        # Use a 1D vector: [radar0, radar1, radar2, radar3, radar4, radar5, radar6, radar7, velocity, angle, difference]
        self.observation_space = spaces.Box(
            low = np.array([0, 0, 0, 0, -1, -3, -1]),
            high = np.array([4, 4, 4, 4, 1, 3, 1]),
            shape = (7,),
            dtype = np.int8
        )

        self.clock = pygame.time.Clock()
        # self.new_img = None
        # self.start_time = None
        # self.distance = None
    
    def initialize_game():
        # start_up_sound.play()

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

        free_spot_index = random.randint(1, 10)
        print(f"Free spot: {free_spot_index}")
        parking_spots.pop(free_spot_index)

        SPOT_RECTANGLES = [pygame.Rect(140.83, 201.5, 75, 100), pygame.Rect(239.16, 201.5, 75, 100), pygame.Rect(337.49, 201.5, 75, 100), pygame.Rect(435.82, 201.5, 75, 100), pygame.Rect(534.15, 201.5, 75, 100), pygame.Rect(140.83, 448.5, 75, 100), pygame.Rect(239.16, 448.5, 75, 100), pygame.Rect(337.49, 448.5, 75, 100), pygame.Rect(435.82, 448.5, 75, 100), pygame.Rect(534.15, 448.5, 75, 100)]

        free_spot_rect = SPOT_RECTANGLES[free_spot_index - 1]     # the rectangle that will turn green when the player parks the car inside it
        FREE_SPOT_BORDER = pygame.image.load(f"parking_game/imgs/free-spot-border-{free_spot_index}.png")       
        FREE_SPOT_BORDER_MASK = pygame.mask.from_surface(FREE_SPOT_BORDER)              

        PARKING_LOT_BORDER = pygame.image.load(f"parking_game/imgs/parking-lot-border-{free_spot_index}.png")
        PARKING_LOT_BORDER_MASK = pygame.mask.from_surface(PARKING_LOT_BORDER)

    # Gym required function (and parameters) to reset the environment
    def reset(self, seed=None, options=None):
        super().reset(seed=seed) # gym requires this call to control randomness and reproduce scenarios.

        # Reset the Parking Optionally, pass in seed control randomness and reproduce scenarios.
        self.car.reset(seed=seed)

        # Construct the observation state:
        # [radar0, radar1, radar2, radar3, radar4, radar5, radar6, radar7, velocity, angle, distance]
        obs = np.array(self.car.discretize_state()).astype(np.int8)
        
        # Additional info to return. For debugging or whatever.
        info = {}

        # Render environment
        # if(self.render_mode=='human'):
        #     self.render()

        # Return observation and info
        return obs, info
    
    # Gym required function (and parameters) to perform an action
    def step(self, action):

        # for event in pygame.event.get():            
        #     if event.type == pygame.QUIT:       # If the user closes the window, the game stops
        #         pygame.quit()
        #         sys.exit()

        # player_car.epsilon = player_car.min_epsilon + (player_car.max_epsilon - player_car.min_epsilon)* np.exp(-player_car.decay_rate * episode)
        self.car.check_radars(PARKING_LOT_BORDER_MASK)
        self.car.move_player(AgentAction(action))           # Perform action
        # self.car.move_player()                            # Change to this to control the car with arrow keys
        terminated, collides, parked = self.car.check_collision()

        # Construct the observation state:
        # [radar0, radar1, radar2, radar3, radar4, radar5, radar6, radar7, velocity, angle, distance]
        obs = np.array((self.car.discretize_state())).astype(np.int8)
    
        # Determine reward and termination
        reward = 0

        if collides:
            reward -= 5000              # punish the car for colliding with an object

        else:
            if parked:
                reward += 200 * (1 / (1 + abs(self.car.vel)))     # reward the car when parked (higher reward when the car is stationary)
            elif self.car.vel == 0:
                reward -= 1000                      # punish the car for standing still when it has not parked   
            if terminated:
                reward += 20000                     # reward the car for parking in the spot for 1 second
        
        # if the car gets closer to the center of the parking spot, give it a reward of 10
        # else if the car gets further from the center of the parking spot, give it a reward of -10
        # else don't give it any reward
        reward += self.car.difference * 10
        # print(f"Reward: {reward}")

        # Additional info to return. For debugging or whatever.
        info = {}

        # Render environment
        if(self.render_mode == 'human'):
            # print(AgentAction(action))
            self.render()

        # Return observation, reward, terminated, truncated (not used), info
        return obs, reward, terminated, False, info
    
    # Gym required function to render environment
    def render(self):
        WIN.blit(PARKING_LOT, (0, 0))

        for index, spot in parking_spots.items():
            WIN.blit(spot[1], (spot[2], spot[3]))
        
        pygame.draw.rect(WIN, free_spot_color, free_spot_rect, 2)
            
        self.car.draw()
        # pygame.draw.circle(WIN, (0, 0, 255), free_spot_rect.center, 3)       # draw the center of the parking spot with blue color
        if intersection is not None:
            pygame.draw.rect(WIN, (255, 0, 0), intersection)            # draw a red rectangle around the point of intersection

        pygame.display.update()

        self.clock.tick(self.car.fps)


class AbstractCar:
    def __init__(self, max_vel, fps):
        self.img = self.IMG
        self.max_vel = max_vel
        self.vel = 0
        # self.angle = random.randint(0, 360)
        # self.x, self.y = self.calculate_START_POS()
        self.acceleration = 0.1
        # self.last_x, self.last_y = self.x, self.y
        # self.rotate_center()
        self.fps = fps
        self.distance = 0
        self.difference = None

    def calculate_START_POS(self):       
        SPAWN_RECTS = [pygame.Rect(2, 2, 651, 95), pygame.Rect(2, 558, 651, 95), pygame.Rect(2, 97, 26, 461), pygame.Rect(627, 97, 26, 461)]        # these are the rectangles where the car can spawn
        car_spawn_index = random.randint(0, 3)
        car_spawn = SPAWN_RECTS[car_spawn_index]
        car_spawn.x += random.randint(0, car_spawn.width)               # randomize the spawn position of the player car
        car_spawn.y += random.randint(0, car_spawn.height)
        print(f"Car spawn: {car_spawn.x}, {car_spawn.y}")
        rotated_image = pygame.transform.rotate(self.img, self.angle)
        new_rect = rotated_image.get_rect(center=self.img.get_rect(topleft= (car_spawn.x, car_spawn.y)).center)
        new_rect.topleft = (car_spawn.x, car_spawn.y)
        print(f"New_rect x: {new_rect.x}, y: {new_rect.y}")

        return new_rect.x + new_rect.width / 2 - self.img.get_width() / 2, new_rect.y + new_rect.height / 2 - self.img.get_height() / 2

    def draw(self):
        global new_img
        # pygame.draw.rect(WIN, (0, 0, 0), new_img[1])                    # draw the new_rect rectangle around the car
        # pygame.draw.circle(WIN, (255, 0, 0), new_img[1].topleft, 5)     # draw the new_rect.x and new_rect.y coordinates with red color
        # pygame.draw.circle(WIN, (0, 0, 255), (self.x, self.y), 5)       # draw the self.x and self.y coordinates with blue color
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
        parked = False
        terminated = False

        if self.collide_map(new_img[1], new_img[2]):
            collision_sound.set_volume(max(min(abs(self.vel * 0.01), 0.02), 0.008))
            # print(f"Volume is: {max(min(abs(self.vel * 0.01), 0.02), 0.008)}")
            # collision_sound.play()
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
        global start_time

        if self.collide_free_spot(new_img[1], new_img[2]):
            # if free_spot_color == (255, 0, 0):         # if the color is red, it means that the car has just parked in the spot, so play the sound
                # green_sound.play()
            free_spot_color = (0, 255, 0)
            if self.vel == 0:                          # if the car is stationary in the spot
                parked = True                          
                if start_time is None:                 # if it just parked, start the timer
                    start_time = time.time()
                elif  time.time() - start_time >= 1:   # else if the car has been stationary for 1 second, stop the game
                    terminated = True
                    return terminated, collides, parked
            else:
                start_time = None                      # if the car is not stationary, reset the timer
        else:
            free_spot_color = (255, 0, 0)
        return terminated, collides, parked
    
    def collide_map(self, new_rect, new_mask):
        global intersection
        offset = (int(new_rect.x), int(new_rect.y))     # offset is the difference between the top left corner of the car image and the top left corner of the window (border mask)
        
        if new_rect.colliderect(GARDEN):                # colliderect() is less computanionally expensive than pixel perfect collision detection using masks. That's why we first check if the rectangles collide.
            # pygame.draw.rect(WIN, (0, 0, 0), new_rect)
            if GARDEN_BORDER_MASK.overlap(new_mask, offset) is not None:   # now we check for pixel perfect collision, because when the car is turning, the new_rect rectangle is bigger than the car image. This leads to false positive collision detetctions when the car is turning around the edges of the garden.
                intersection = new_rect.clip(GARDEN)                       # returns a new rectangle that represents the intersection of the two rectangles.
                print(f"collision with garden")
                return True
        elif new_rect.colliderect(TOP_RECT):
                intersection = None
                print(f"collision with top rect")
                print(f"x: {new_rect.x}, y: {new_rect.y}")
                return True
        elif new_rect.colliderect(BOTTOM_RECT):
                intersection = None
                print(f"collision with bottom rect")
                print(f"x: {new_rect.x}, y: {new_rect.y}")
                return True
        elif new_rect.colliderect(LEFT_RECT):
                intersection = None
                print(f"collision with left rect")
                print(f"x: {new_rect.x}, y: {new_rect.y}")
                return True
        elif new_rect.colliderect(RIGHT_RECT):
                intersection = None
                print(f"collision with right rect")
                print(f"x: {new_rect.x}, y: {new_rect.y}")
                return True
        for index, spot in parking_spots.items():
            if new_rect.colliderect(spot[0]):
                if PARKING_LOT_BORDER_MASK.overlap(new_mask, offset) is not None: 
                    intersection = new_rect.clip(spot[0])
                    print(f"collision with car {index}")
                    print(f"x: {new_rect.x}, y: {new_rect.y}")
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
        print(f"{self.vel:.2f}")
        self.vel = -self.vel                            # reverse the direction of the car, so that it exits from colliding 
        if round(self.vel, 2) == 0.00:         # this was used for when the car was stuck colliding while having velocity = 0, the game would crash
            self.vel = -0.1       # however I think this is not necessary anymore, because the car will always have a velocity different from 0 (you can not press the up arrow key and the down arrow key at the same time)
        counter = 0
        while True:
            print(f"{self.vel:.2f}")
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
        self.vel = max(self.vel - self.acceleration, -self.max_vel * 3/4)   # max because these are negative values, we are still choosing the lower speed
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
        self.x, self.y = self.calculate_START_POS()
        self.last_x, self.last_y = self.x, self.y
        self.rotate_center()

    def reset(self, seed = None):
        random.seed(seed)
        self.return_to_map()
        ParkingGameEnv.initialize_game()
        self.check_radars(PARKING_LOT_BORDER_MASK)

    def check_radars(self, game_map):
        self.radars = [[(0, 0), 0] for _ in range(4)]
        degrees = [75, 105, 255, 285]
        # degrees = [0, 45, 90, 135, 180, 225, 270, 315]
        step_size = 20
        for i in range(4):
            length = 45                 # the starting length of the radar depends on its angle (because we don't want the radar to start checking inside the car)
            offset = 40                 # offset is the distance from the center of the car to the edge of the car image. It depends on the angle of the radar. We use it to calculate the real distance of the radar.
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
        previous_distance = self.distance 
        for radar in self.radars:
            radar[1] = radar[1] // 50        # The discretized radar has 5 bins, in range [0, 4]
        # print(f"Discretized radar 1: {self.radars[0][1]}") 
        discrete_vel = 1 if  self.vel > 0 else -1 if self.vel < 0 else 0       # The discretized velocity has 3 bins, in range [-1, 1]
        # print(f"Self.vel: {self.vel:.2f}    discrete_vel: {discrete_vel}")       
        discrete_angle = -(-math.floor((round(math.sin(math.radians(self.angle)), 1)  * 10) / 2) //2)  if math.sin(math.radians(self.angle)) > 0 else math.ceil((round(math.sin(math.radians(self.angle)), 1)  * 10) / 2) // 2   # The discretized angle has 7 bins, in range [-3, 3]
        # print(f"Self.angle: {self.angle:.2f}    discrete_angle: {discrete_angle}") 
        self.distance = int(math.sqrt(math.pow(self.center[0] - free_spot_rect.centerx, 2) + math.pow(self.center[1] - free_spot_rect.centery, 2)))     # the distance of the car to the center of the parking spot
        # distance_discrete = self.distance // 100 + 9 if self.distance >= 100 else self.distance // 10          # The discretized distance has 17 bins, in range [0, 16]
        # print(f"Distance: {distance}    Distance discrete: {distance_discrete}")
        self.difference = 1 if previous_distance - self.distance > 0 and abs(self.vel) > 0.5  else -1 if previous_distance - self.distance < 0 else 0    # the difference between the previous distance and the current distance has 3 bins, in range [-1, 1]
        # print(f"Difference: {self.difference}")
        
        return self.radars[0][1], self.radars[1][1], self.radars[2][1], self.radars[3][1], discrete_vel, discrete_angle, self.difference


class PlayerCar(AbstractCar):           # the player car will have additional methods for moving using the arrow keys
    IMG = RED_CAR[0]

    def move_player(self):
        self.player_action = ""
        keys = pygame.key.get_pressed()
        throttling = False   
        self.img = RED_CAR[0]           # the car image is set to the default image, so that it does not rotate when the player is not pressing the left or right arrow key     
        if not (keys[pygame.K_LEFT] and keys[pygame.K_RIGHT]): 
            if keys[pygame.K_LEFT]:                 # Keyboard ghosting is a hardware issue where certain combinations of keys cannot be detected simultaneously due to the design of the keyboard.
                    self.rotate(left=True)          # Due to this limitation of keyboard, we can only detect two arrow key presses at a time. This means that if the player is pressing the left and the right arrow key
                    self.img = pygame.transform.flip(RED_CAR[1], True, False)     # and then presses the up arrow key, the car will not move, as this third key press will not be detected.                                
                    self.player_action = "LEFT "
            elif keys[pygame.K_RIGHT]:                
                    self.rotate(right=True)
                    self.img = RED_CAR[1]   
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
    IMG = RED_CAR[0]
    
    def move_player(self, agent_action):
        throttling = False   
        self.img = RED_CAR[0]           # the car image is set to the default image, so that it does not rotate when the player is not pressing the left or right arrow key     
        if agent_action == AgentAction.LEFT or agent_action == AgentAction.DOWN_LEFT or agent_action == AgentAction.UP_LEFT:                 # Keyboard ghosting is a hardware issue where certain combinations of keys cannot be detected simultaneously due to the design of the keyboard.
                self.rotate(left=True)          
                self.img = pygame.transform.flip(RED_CAR[1], True, False)    
        elif agent_action == AgentAction.RIGHT or agent_action == AgentAction.DOWN_RIGHT or agent_action == AgentAction.UP_RIGHT:                
                self.rotate(right=True)
                self.img = RED_CAR[1]                                # we change the car img to the one that the wheels are turning
        if agent_action == AgentAction.UP or agent_action == AgentAction.UP_LEFT or agent_action == AgentAction.UP_RIGHT:
            throttling = True
            self.move_forward()
        elif agent_action == AgentAction.DOWN or agent_action == AgentAction.DOWN_LEFT or agent_action == AgentAction.DOWN_RIGHT:
            throttling = True
            self.move_backward()

        if not throttling and self.vel != 0:                # if the player is not stepping on the gas, reduce the speed 
            self.reduce_speed()


class AgentAction(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    UP_LEFT = 4
    UP_RIGHT = 5
    DOWN_LEFT = 6
    DOWN_RIGHT = 7
    NOTHING = 8



# For unit testing with random actions
# if __name__=="__main__":
#     env = gym.make('parking-game-v0', render_mode='human')

#     # Use this to check our custom environment
#     # print("Check environment begin")
#     # check_env(env.unwrapped)
#     # print("Check environment end")

#     # Reset environment
#     obs = env.reset(seed=8)[0]      # fixed spawns each time

#     # Take some random actions
#     while(True):
#         rand_action = env.action_space.sample()
#         obs, reward, terminated, _, _ = env.step(rand_action)
#         print(f"Observations: {obs}   Action: {AgentAction(rand_action).name}   Reward: {reward}   Terminated: {terminated}")
#         # print(f"Observations: {obs}   Action: {env.unwrapped.car.player_action}   Reward: {reward}   Terminated: {terminated}")     # for when the car is controlled using the arrow keys

#         if(terminated):
#             obs = env.reset()[0]

# Number of steps per episode taken by the agent to park.
# Since we operate at 20 fps, the agent chooses 20 actions per second. The car
# can always be parked in less than 30 seconds, so we will allow max 20 x 30 = 600 steps.
max_steps = 600

# Train using Q-Learning (either from scratch or continue training by loading Q Table from file)
def train_q(total_episodes, render=False, episodes_previously_trained=0):

    env = gym.make('parking-game-v0', render_mode='human' if render else None)

    if episodes_previously_trained > 0:
        q = np.load('parking_game/Q-tables/parking_q_30000.npy')   # CHANGE THIS TO THE LAST EPISODE NUMBER
        epsilon = 1                                  # CHANGE THIS TO PREVIOUS EPSILON VALUE
    
    else:
        # Initialize the Q Table, a 8D array of zeros.
        q = np.zeros((5, 5, 5, 5, 3, 7, 3, 9), dtype=np.float16)        # 2 Bytes per element

        # Hyperparameters
        epsilon = 1.0   # 1 = 100% random actions
    
    # min_epsilon = 0.05
    # decay_rate = 0.0005  # the higher the decay rate, the faster the epsilon will decrease and the agent will start to exploit more than explore
    alpha = 1   # learning rate, 1 = 100% weight on new information, it is the optimal value since the environment is deterministic
    gamma = 0.9   # discount rate. Near 0: more weight/reward placed on immediate state. Near 1: more on future state. Some choose 0.95 or 0.99.
    # count = 0

    episode_rewards = []
    episode_successes = []      # 1 if car parked, 0 if not
    # episode_times = []

    start_time = time.time()

    for episode in range(episodes_previously_trained, total_episodes):
        # episode_start_time = time.time()
        # count += 1
        # print(f'Train Episode {episode+1}')

        
        state = env.reset(seed=32)[0]          # Reset environment at the beginning of episode
        # print(f"State: {state}")
        terminated = False
        total_reward = 0
        episode_successes.append(0)

        
        for step in range(max_steps):       # Agent controls the car until it parks or max steps reached

            for event in pygame.event.get():            
                if event.type == pygame.QUIT:       # If the user closes the window, the game stops
                    if episode > 100:
                        np.save(f"parking_game/Q-tables/parking_q_{episode}.npy", q)
                        print_stats(start_time, epsilon, episode_rewards, episode_successes, episodes_previously_trained, episode)
                        plot_graphs(episode_rewards, train=True)
                    pygame.quit()
                    sys.exit()

            # Select action based on epsilon-greedy
            if random.random() < epsilon:
                # select random action
                action = env.action_space.sample()
            else:                
                # Convert state of [1,2,3,4,5,6,7] into (1,2,3,4,5,6,7), use this to index into the 7th dimension of the 8D array.
                q_state_idx = tuple(state) 

                # select best action
                action = np.argmax(q[q_state_idx])
            
            # Perform action
            new_state,reward,terminated,_,_ = env.step(action)
            total_reward += reward

            # Convert state of [1,2,3,4,5,6,7] and action of [1] into (1,2,3,4,5,6,7,1), use this to index into the 7th dimension of the 8D array.
            q_state_action_idx = tuple(state) + (action,)

            # Convert new_state of [1,2,3,4,5,6,7] into (1,2,3,4,5,6,7), use this to index into the 7th dimension of the 8D array.
            q_new_state_idx = tuple(new_state)

            # Update Q-Table
            # new_value = int(q[q_state_action_idx] + alpha * (reward + gamma * np.max(q[q_new_state_idx]) - q[q_state_action_idx]))
            # q[q_state_action_idx] = -128 if new_value < -128 else 127 if new_value > 127 else new_value
            q[q_state_action_idx] = q[q_state_action_idx] + alpha * (reward + gamma * np.max(q[q_new_state_idx]) - q[q_state_action_idx])

            if terminated:
                episode_successes[-1] = 1
                break

            # Update current state
            state = new_state

        # Decrease epsilon
        epsilon = max(epsilon - 1/total_episodes, 0.05)
        # epsilon = min_epsilon + (1 - min_epsilon) * np.exp(-decay_rate * count)

        episode_rewards.append(total_reward)

        if episode % 1000 == 0:     # Save Q-Table every 1000 episodes
            # f = open(f"parking_game/parking_q_{episode}.pkl","wb")
            # pickle.dump(q, f)
            # f.close()
            np.save(f"parking_game/Q-tables/parking_q_{episode}.npy", q)

        # episode_times.append(time.time() - episode_start_time)


    # print the episode mean time in minutes and seconds
    # print(f"\nEpisode mean time: {np.mean(episode_times)//60:.0f} minutes, {np.mean(episode_times)%60:.2f} seconds")    

    env.close()

    print_stats(start_time, epsilon, episode_rewards, episode_successes, episodes_previously_trained, total_episodes) 
    plot_graphs(episode_rewards, train=True)    # Graph rewards


def print_stats(start_time, epsilon, episode_rewards, episode_successes, episodes_previously_trained, episodes_currently_trained):
    training_time = time.time() - start_time
    print(f"\nTraining time: {training_time//3600:.0f} hours, {(training_time%3600)//60:.0f} minutes, {training_time%60:.2f} seconds")

    print(f"\nEpsilon: {epsilon}")

    print("\nMean reward per thousand episodes")
    for i in range((episodes_currently_trained - episodes_previously_trained) //1000):
        print(f"{episodes_previously_trained + (i*1000)}-{episodes_previously_trained + ((i+1)*1000)}: mean episode reward: {np.mean(episode_rewards[i*1000:(i+1)*1000])}")

    print("\nMean success rate per thousand episodes")
    for i in range((episodes_currently_trained - episodes_previously_trained) //1000):
        print(f"{episodes_previously_trained + (i*1000)}-{episodes_previously_trained + ((i+1)*1000)}: mean episode success: {(np.mean(episode_successes[i*1000:(i+1)*1000]) * 100):.2f} %")

def plot_graphs(episode_rewards, train=False):
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)        # standard deviation
    plt.plot(episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(f'Q-Learning Rewards (Mean: {mean_reward:.2f}, +/- {std_reward:.2f})')
    if train:
        plt.savefig('parking_game/parking_q_rewards-train.png')
    else:
        plt.savefig('parking_game/parking_q_rewards-test.png')
    plt.show()


def test_q(test_episodes, episodes_trained, render=True):
    
    env = gym.make('parking-game-v0', render_mode='human' if render else None)

    q = np.load('parking_game/Q-tables/parking_q_' + str(episodes_trained) + '.npy')  # load Q Table from file

    episode_rewards = []
    successful_episodes = 0

    for episode in range(test_episodes):
        print(f'Test Episode {episode}')

        state = env.reset(seed=32)[0]          # Reset environment at the beginning of episode
        terminated = False
        total_reward = 0

        
        for step in range(max_steps):   # Agent controls the car until it parks or max steps reached

            for event in pygame.event.get():            
                if event.type == pygame.QUIT:       # If the user closes the window, the game stops
                    pygame.quit()
                    sys.exit()

            # Convert state of [1,2,3,4,5,6,7,8,9,10,11] into (1,2,3,4,5,6,7,8,9,10,11), use this to index into the 11th dimension of the 12D array.
            q_state_idx = tuple(state)
            print(f"State: {state}", end=" ") 

            # select best action
            action = np.argmax(q[q_state_idx])
            
            # Perform action
            state,reward,terminated,_,_ = env.step(action)
            print(f"Reward: {reward}")
            total_reward += reward

            if terminated:
                successful_episodes += 1
                break

        print(f'Test Episode {episode} Reward: {total_reward}')
        episode_rewards.append(total_reward)

    env.close()

    # Graph success rate
    print(f'Agent {episodes_trained} Success Rate: {successful_episodes}/{test_episodes}')
    
    plot_graphs(episode_rewards)    # Graph rewards



if __name__ == '__main__':

    # Train/test using Q-Learning
    # train_q(60001, render=False, episodes_previously_trained=30001)
    test_q(10, 50000, render=True)