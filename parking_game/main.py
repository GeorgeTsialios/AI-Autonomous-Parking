import pygame
import time
import math
from utils import scale_image, blit_rotate_center

PARKING_LOT = scale_image(pygame.image.load("parking_game/imgs/parking-lot.png"), 1)

#PARKING_LOT_BORDER = scale_image(pygame.image.load("parking_game/imgs/PARKING_LOT-border.png"), 0.85)
#PARKING_LOT_BORDER_MASK = pygame.mask.from_surface(PARKING_LOT_BORDER)

#FINISH = pygame.image.load("parking_game/imgs/finish.png")
#FINISH_MASK = pygame.mask.from_surface(FINISH)
# FINISH_POSITION = (130, 250)

RED_CAR = scale_image(pygame.image.load("parking_game/imgs/red-car2.png"), 40/162)            # factor is equal to desired width of car / actual width of image
YELLOW_CAR = scale_image(pygame.image.load("parking_game/imgs/yellow-car.png"), 40/162)       # this way all cars have the same width (40px) 
PINK_CAR = scale_image(pygame.image.load("parking_game/imgs/pink-car.png"), 40/162)
GREEN_CAR = scale_image(pygame.image.load("parking_game/imgs/green-car.png"), 40/163)
PURPLE_CAR = scale_image(pygame.image.load("parking_game/imgs/purple-car.png"), 40/164)
# WHITE_CAR = scale_image(pygame.image.load("parking_game/imgs/white-car-old.png"), 40/38)


WIDTH, HEIGHT = PARKING_LOT.get_width(), PARKING_LOT.get_height()
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
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
        if left:
            self.angle += self.rotation_vel
        elif right:
            self.angle -= self.rotation_vel

    def draw(self, win):
        blit_rotate_center(win, self.img, (self.x, self.y), self.angle)

    def move_forward(self):
        self.vel = min(self.vel + self.acceleration, self.max_vel)
        self.move()

    def move_backward(self):    # move backward with half the max speed
        self.vel = max(self.vel - self.acceleration, -self.max_vel/2)   # max becasue these are negative values, we are still choosing the lower speed
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


class PlayerCar(AbstractCar):
    # IMG = WHITE_CAR
    # IMG = GREEN_CAR
    # IMG = YELLOW_CAR
    # IMG = PURPLE_CAR
    # IMG = PINK_CAR
    IMG = RED_CAR
    START_POS = (400, 200)

    def reduce_speed(self):
        self.vel = max(self.vel - self.acceleration / 2, 0)     # friction is half the acceleration
        self.move()

    def bounce(self):
        self.vel = -self.vel
        self.move()


def draw(win, images, player_car):
    for img, pos in images:
        win.blit(img, pos)

    player_car.draw(win)
    pygame.display.update()


def move_player(player_car):
    keys = pygame.key.get_pressed()
    throttling = False        

    if keys[pygame.K_LEFT]:
        player_car.rotate(left=True)
    if keys[pygame.K_RIGHT]:
        player_car.rotate(right=True)
    if keys[pygame.K_UP]:
        throttling = True
        player_car.move_forward()
    if keys[pygame.K_DOWN]:
        throttling = True
        player_car.move_backward()

    if not throttling:                # if the player is not stepping on the gas, reduce the speed 
        player_car.reduce_speed()


run = True
clock = pygame.time.Clock()
FPS = 20
images = [(PARKING_LOT, (0, 0)) ]
player_car = PlayerCar(8, 8)
counter = 1

while run:
    clock.tick(FPS)

    draw(WIN, images, player_car)

    for event in pygame.event.get():            
        if event.type == pygame.QUIT:       # If the user closes the window, the game stops
            run = False
            break

    move_player(player_car)

    # if player_car.collide(PARKING_LOT_BORDER_MASK) != None:
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
