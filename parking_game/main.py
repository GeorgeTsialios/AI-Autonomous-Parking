# This is going to be a parking game where the player has to park the car in the parking lot. The player car will be controlled by the arrow keys.
# The cars' movement should resemble closely real-life physics in terms of acceleration, deceleration, and turning.
# There will be 10 parking spots, but only one available for the player, while the rest will be occupied by other cars. The available parking spot 
# will contain a rectangle, which will turn green once the player parks the car inside it. The available parking spot will be randomly chosen at the
# beginning of each game. Also, the player car will spawn at a random position at the beginning of each game.
# There will be collision detection between the player car and the other cars, as well as the parking lot borders and a garden (depicted as a rectangle
# in the middle part of the parking lot).

import pygame
import time
import math
from utils import scale_image, blit_rotate_center

PARKING_LOT = pygame.image.load("parking_game/imgs/parking-lot.png")

RED_CAR = scale_image(pygame.image.load("parking_game/imgs/red-car2.png"), 40/162)            # factor is equal to desired width of car / actual width of image
YELLOW_CAR = scale_image(pygame.image.load("parking_game/imgs/yellow-car.png"), 40/162)       # this way all cars have the same width (40px) 
PINK_CAR = scale_image(pygame.image.load("parking_game/imgs/pink-car.png"), 40/162)
GREEN_CAR = scale_image(pygame.image.load("parking_game/imgs/green-car.png"), 40/163)
PURPLE_CAR = scale_image(pygame.image.load("parking_game/imgs/purple-car.png"), 40/164)
# WHITE_CAR = scale_image(pygame.image.load("parking_game/imgs/white-car-old.png"), 40/38)

GARDEN = pygame.Rect(125, 325, 500, 100)

WIN_WIDTH, WIN_HEIGHT = PARKING_LOT.get_width(), PARKING_LOT.get_height()
WIN = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
pygame.display.set_caption("Parking Game!")


class AbstractCar:
    def __init__(self, max_vel, rotation_vel):
        self.img = self.IMG
        self.max_vel = max_vel
        self.vel = 0
        self.rotation_vel = rotation_vel
        self.angle = 0
        self.x, self.y = self.START_POS
        self.acceleration = 0.1

    def rotate(self, left=False, right=False):
        if abs(self.vel) > 0:        # if the car is moving, it can rotate
            if left:
                self.angle += self.rotation_vel * self.vel
            elif right:
                self.angle -= self.rotation_vel * self.vel

    def draw(self, win):
      if  blit_rotate_center(win, self.img, (self.x, self.y), self.angle):
          self.bounce()
    
    def bounce(self):
        self.vel = -self.vel
        self.move()
        while blit_rotate_center(WIN, self.img, (self.x, self.y), self.angle):
          self.move()
        self.vel = 0

    def move_forward(self):
        self.vel = min(self.vel + self.acceleration, self.max_vel)
        self.move()

    def move_backward(self):    # move backward with half the max speed
        self.vel = max(self.vel - self.acceleration, -self.max_vel/2)   # max because these are negative values, we are still choosing the lower speed
        self.move()

    def move(self):
        angle_radians = math.radians(self.angle)
        vertical_distance = math.cos(angle_radians) * self.vel       # the distance to move in the y direction
        horizontal_distance = math.sin(angle_radians) * self.vel     # the distance to move in the x direction

        self.y -= vertical_distance
        self.x -= horizontal_distance

    def collide(self, mask, x=0, y=0):
        ''' 
        Function that ckecks for pixel perfect collision between the car and the PARKING_LOT.
        It returns the point of intersection if there is a collision, otherwise it returns None.
        '''
        car_mask = pygame.mask.from_surface(self.img)
        offset = (int(self.x - x), int(self.y - y))
        poi = mask.overlap(car_mask, offset)        # point of intersection
        return poi

    def reset(self):
        self.x, self.y = self.START_POS
        self.angle = 0
        self.vel = 0
    
    def reduce_speed(self):
        if self.vel > 0:
            self.vel = max(self.vel - self.acceleration / 2, 0)     # friction is half the acceleration
        else:
            self.vel = min(self.vel + self.acceleration / 2, 0)
        self.move()


class PlayerCar(AbstractCar):
    # IMG = WHITE_CAR
    # IMG = GREEN_CAR
    # IMG = YELLOW_CAR
    # IMG = PURPLE_CAR
    # IMG = PINK_CAR
    IMG = RED_CAR
    START_POS = (400, 100)

    def move_player(self):
        keys = pygame.key.get_pressed()
        throttling = False        

        if keys[pygame.K_LEFT]:
                self.rotate(left=True)
        if keys[pygame.K_RIGHT]:
                self.rotate(right=True)
        if keys[pygame.K_UP]:
            throttling = True
            self.move_forward()
        if keys[pygame.K_DOWN]:
            throttling = True
            self.move_backward()

        if not throttling and self.vel !=0:                # if the player is not stepping on the gas, reduce the speed 
            self.reduce_speed()


def draw_window(win, player_car):
    win.blit(PARKING_LOT, (0, 0))

    player_car.draw(win)
    pygame.display.update()


run = True
clock = pygame.time.Clock()
FPS = 20
player_car = PlayerCar(8, 1)

while run:
    clock.tick(FPS)

    for event in pygame.event.get():            
        if event.type == pygame.QUIT:       # If the user closes the window, the game stops
            run = False
            break

    player_car.move_player()
    draw_window(WIN, player_car)

    # if player_car.img.get_rect(topleft=(player_car.x, player_car.y)).colliderect(GARDEN):
    #     print(f"collision no: {counter}")
    #     counter += 1
    #     player_car.bounce()

    # finish_poi_collide = player_car.collide(FINISH_MASK, *FINISH_POSITION) 
    # if finish_poi_collide != None:
    #     if finish_poi_collide[1] == 0:
    #         player_car.bounce()             # if the car hits the finish line from the top, bounce
    #     else:
    #         player_car.reset()
    #         print("finish lap")


pygame.quit()
