# This is going to be a parking game where the player has to park the car in the parking lot. The player car will be controlled by the arrow keys.
# The car's movement should resemble closely real-life physics in terms of acceleration, deceleration, and turning.
# There will be 10 parking spots, but only one free for the player, while the rest will be occupied by other cars. The free parking spot 
# will contain a rectangle, which will turn green once the player places the car inside it. The game resets once the car is stationary inside it for 2 seconds.
# The free parking spot will be randomly chosen at the beginning of each game. Also, the player car will spawn at a random position at the beginning of each game.
# There will be collision detection between the player car and the other cars, as well as the parking lot borders and a garden (depicted as a rectangle
# in the middle part of the parking lot). The car will have 8 depth sensors (radars), which will detect the distance to the nearest object (or window borders) in 8 directions.
# The radars will be drawn on the screen as lines.

import pygame
import time
import math
import random

# random.seed(4)

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

def initialize_game():
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

    free_spot_index = random.randint(1, 10)
    print(f"Free spot: {free_spot_index}")
    parking_spots.pop(free_spot_index)

    SPOT_RECTANGLES = [pygame.Rect(140.83, 201.5, 75, 100), pygame.Rect(239.16, 201.5, 75, 100), pygame.Rect(337.49, 201.5, 75, 100), pygame.Rect(435.82, 201.5, 75, 100), pygame.Rect(534.15, 201.5, 75, 100), pygame.Rect(140.83, 448.5, 75, 100), pygame.Rect(239.16, 448.5, 75, 100), pygame.Rect(337.49, 448.5, 75, 100), pygame.Rect(435.82, 448.5, 75, 100), pygame.Rect(534.15, 448.5, 75, 100)]

    free_spot_rect = SPOT_RECTANGLES[free_spot_index - 1]     # the rectangle that will turn green when the player parks the car inside it
    FREE_SPOT_BORDER = pygame.image.load(f"parking_game/imgs/free-spot-border-{free_spot_index}.png")       
    FREE_SPOT_BORDER_MASK = pygame.mask.from_surface(FREE_SPOT_BORDER)              

    PARKING_LOT_BORDER = pygame.image.load(f"parking_game/imgs/parking-lot-border-{free_spot_index}.png")
    PARKING_LOT_BORDER_MASK = pygame.mask.from_surface(PARKING_LOT_BORDER)


class AbstractCar:
    def __init__(self, max_vel, rotation_vel):
        self.img = self.IMG
        self.max_vel = max_vel
        self.vel = 0
        self.rotation_vel = rotation_vel
        self.angle = random.randint(0, 360)
        self.x, self.y = self.calculate_START_POS()
        self.acceleration = 0.1
        self.last_x, self.last_y = self.x, self.y
        self.rotate_center()
        self.initialize_radars(PARKING_LOT_BORDER_MASK)

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
        # pygame.draw.circle(WIN, (0, 255, 0), self.center, 5)       # draw the center of the car with green color
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

        if self.collide_map(new_img[1], new_img[2]):
            collision_sound.set_volume(max(min(abs(self.vel * 0.01), 0.02), 0.008))
            # print(f"Volume is: {max(min(abs(self.vel * 0.01), 0.02), 0.008)}")
            collision_sound.play()
            # pygame.display.update()
            self.bounce()
        
        elif new_img[1].x < 0 or new_img[1].x > WIN_WIDTH or new_img[1].y < 0 or new_img[1].y > WIN_HEIGHT:       # if the car goes out of the window, return_to_map it
            print(f"out of bounds - x: {new_img[1].x}, y: {new_img[1].y}")
            intersection =  None
            self.return_to_map()

        else:
             self.last_x, self.last_y = self.x, self.y      # save the last, safe position of the car (where it does not collide with anything)
             intersection = None

        global free_spot_color
        global run
        global start_time

        if self.collide_free_spot(new_img[1], new_img[2]):
            if free_spot_color == (255, 0, 0):         # if the color is red, it means that the car has just parked in the spot, so play the sound
                green_sound.play()
            free_spot_color = (0, 255, 0)
            if self.vel == 0:                          # if the car is stationary in the spot
                if start_time is None:                 # if it just parked, start the timer
                    start_time = time.time()
                elif  time.time() - start_time >= 2:   # else if the car has been stationary for 2 seconds, stop the game
                    self.reset()
            else:
                start_time = None                      # if the car is not stationary, return_to_map the timer
        else:
            free_spot_color = (255, 0, 0)
    
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
        if self.vel == 0:         # this was used for when the car was stuck colliding while having velocity = 0, the game would crash
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
        if abs(self.vel) > 0:        # if the car is moving, it can rotate
            if left:
                self.angle += self.rotation_vel * self.vel
            elif right:
                self.angle -= self.rotation_vel * self.vel

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

    def reset(self):
        self.return_to_map()
        initialize_game()
        self.initialize_radars(PARKING_LOT_BORDER_MASK)

    def initialize_radars(self, game_map):
        self.radars = [0, 0, 0, 0, 0, 0, 0, 0]
        degree = 0
        for i in range(8):
            length = 0
            x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * length)
            y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * length)

            while length < 200:
                test_x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * (length + 1))
                test_y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * (length + 1))
                if game_map.get_at((test_x, test_y)) != 0:
                    break
                length = length + 1
                x = test_x
                y = test_y

            # Calculate Distance To Border And Append To Radars List
            # dist = int(math.sqrt(math.pow(x - self.center[0], 2) + math.pow(y - self.center[1], 2)))
            self.radars[i] = [(x, y), length]
            degree += 45

    def check_radars(self, game_map):
        degree = 0
        step_size = 20
        for i in range(8):
            collide = False
            length = self.radars[i][1] 
            
            x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * length)   # the previous length of the radar
            y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * length)

            if (x > 0 and x < 750) and (y > 0 and y < 750) and game_map.get_at((x, y)) == 0:  # if the radar is not colliding with an object at its length point
                # we need to check some previous points (with step size = 20), to check if the radar is colliding with an object at a closer distance
                test_length = length
                while test_length - step_size > 0:
                    test_length = test_length - step_size
                    test_x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * test_length)
                    test_y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * test_length)
                    if (test_x > 0 and test_x < 750) and (test_y > 0 and test_y < 750) and game_map.get_at((test_x, test_y)) != 0:
                        length = test_length
                        collide = True
                        break
                if not collide:        # if the radar is not colliding with anything, increase its length until it gets to 100 or collides with something     
                    while length < 200:        
                        # print(f"Radar {i} color: ", game_map.get_at((x, y)))
                        # length = length + 1
                        test_x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * (length + 1))
                        test_y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * (length + 1))
                        if (x > 0 and x < 750) and (y > 0 and y < 750) and game_map.get_at((test_x, test_y)) != 0:
                            break
                        length = length + 1
                        x = test_x
                        y = test_y
            else:
                collide = True

            if collide:         # if the radar is colliding with something, decrease its length until it gets to 0 or stops colliding with something
                while length > 0:          # if the radar is colliding with something, decrease its length until it gets to 0 or stops colliding with something
                    # print(f"Radar {i} color: ", game_map.get_at((x, y)))
                    length = length - 1
                    x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * length)
                    y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * length)
                    if (x > 0 and x < 750) and (y > 0 and y < 750) and game_map.get_at((x, y)) == 0:
                        break

            # dist = int(math.sqrt(math.pow(x - self.center[0], 2) + math.pow(y - self.center[1], 2)))      # Calculate Distance To Border And Append To Radars List
            self.radars[i] = [(x, y), length]
            degree += 45


class PlayerCar(AbstractCar):           # the player car will have additional methods for moving using the arrow keys
    IMG = RED_CAR[0]

    def move_player(self):
        keys = pygame.key.get_pressed()
        throttling = False   
        self.img = RED_CAR[0]           # the car image is set to the default image, so that it does not rotate when the player is not pressing the left or right arrow key     
        if keys[pygame.K_LEFT]:                 # Keyboard ghosting is a hardware issue where certain combinations of keys cannot be detected simultaneously due to the design of the keyboard.
                self.rotate(left=True)          # Due to this limitation of keyboard, we can only detect two arrow key presses at a time. This means that if the player is pressing the left and the right arrow key
                self.img = pygame.transform.flip(RED_CAR[1], True, False)     # and then presses the up arrow key, the car will not move, as this third key press will not be detected.                                
        if keys[pygame.K_RIGHT]:                
                self.rotate(right=True)
                self.img = RED_CAR[1]
        if not (keys[pygame.K_UP] and keys[pygame.K_DOWN]):          # if both keys are pressed, the car should not move
            if keys[pygame.K_UP]:
                throttling = True
                self.move_forward()
            if keys[pygame.K_DOWN]:
                throttling = True
                self.move_backward()

        if not throttling and self.vel !=0:                # if the player is not stepping on the gas, reduce the speed 
            self.reduce_speed()


def draw_window(player_car):
    WIN.blit(PARKING_LOT, (0, 0))

    for index, spot in parking_spots.items():
        WIN.blit(spot[1], (spot[2], spot[3]))
    
    pygame.draw.rect(WIN, free_spot_color, free_spot_rect, 2)
        
    player_car.draw()
    if intersection is not None:
        pygame.draw.rect(WIN, (255, 0, 0), intersection)            # draw a red rectangle around the point of intersection

    pygame.display.update()


run = True
clock = pygame.time.Clock()
FPS = 20
initialize_game()
player_car = PlayerCar(6, 1)
new_img = None
start_time = None


while run:
    clock.tick(FPS)

    for event in pygame.event.get():            
        if event.type == pygame.QUIT:       # If the user closes the window, the game stops
            run = False
            break

    player_car.move_player()
    player_car.check_collision()
    player_car.check_radars(PARKING_LOT_BORDER_MASK)
    draw_window(player_car)


pygame.quit()
