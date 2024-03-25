import pygame
import time
import random


pygame.font.init()      # Initialize the font module, essential for rendering text on the screen

WIDTH, HEIGHT = 1000, 700                       # Set the width and height of the window
WIN = pygame.display.set_mode((WIDTH, HEIGHT))  # Create a window
pygame.display.set_caption("Space Dodge")       # Set the title of the window

BG = pygame.transform.scale(pygame.image.load("Assets/bg.jpeg"), (WIDTH, HEIGHT))  # Load the background image and scale it to the window size


# print(pygame.font.get_fonts())

FONT = pygame.font.SysFont("gadugi", 30)    # Select text font and size


class Player:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.vel = 5

    def draw(self):
        pygame.draw.rect(WIN, "red", pygame.Rect(self.x, self.y, self.width, self.height))

    def move(self):
        keys = pygame.key.get_pressed()     # pygame.key.get_pressed() returns a dictionary with the state of all keys
        if keys[pygame.K_LEFT] and self.x - self.vel >= 0:             # the 2nd condition ensures that the player doesn't go out of the window
            self.x -= self.vel
        if keys[pygame.K_RIGHT] and self.x + self.vel + self.width <= WIDTH:            # the 2nd condition ensures that the player doesn't go out of the window
            self.x += self.vel

my_player = Player(WIDTH/2 - 40/2, HEIGHT - 60, 40, 60)


class Star(Player):
    star_count = 0                              # Counter to keep track of time (up to 2000ms)
    stars = []                                  # Active stars
    def __init__(self, x, y, width, height):
        super().__init__(x, y, width, height)
        self.vel = 3
        Star.stars.append(self)

    def move(self):
        self.y += self.vel
        if self.y > HEIGHT:     # If the top of the star goes below the window, remove it
            Star.stars.remove(self)
        
    def check_collision(self, hit):    
        if self.y + self.height >= my_player.y and pygame.Rect(self.x, self.y, self.width, self.height).colliderect(pygame.Rect(my_player.x, my_player.y, my_player.width, my_player.height)): # 1st condition checks if the bottom of the star is below the player (only then can the player collide with the star and thus we have to check the 2nd condition)
            hit = True
        return hit

    def draw():
        for star in Star.stars:
            pygame.draw.rect(WIN, "white", pygame.Rect(star.x, star.y, star.width, star.height))


def draw(elapsed_time):      # updates the window, is called every frame (draw != move, draw is for updating the visuals (uses x,y), move is for updating the postion x,y)
    WIN.blit(BG, (0, 0))     # Draw the background image

    time_text = FONT.render(f"Time: {elapsed_time:.1f}s", 1, "white")   # 1 is the anti-aliasing level, keep it 1 for better quality
    WIN.blit(time_text, (10, 10))           # Draw the time text in its position (x,y)
                                            # blit is called for images and text, draw is called for shapes
    my_player.draw()
    Star.draw()

    pygame.display.update()                 # Essential for the changes to appear on screen


def hit_draw():                    # Draw function that is called when the player hits a star
    lost_text = FONT.render("You Lost!", 1, "white")
    WIN.blit(lost_text, (WIDTH/2 - lost_text.get_width()/2, HEIGHT/2 - lost_text.get_height()/2))   # Center the text
    pygame.display.update()
    pygame.time.delay(2000)


def main():
    run = True
    hit = False

    clock = pygame.time.Clock()     # Create a clock object to control the frame rate
    start_time = time.time()
    elapsed_time = 0
    time_interval = 2000       # Every time_interval ms (at the beginning 2000ms), add 3 stars          


    while run:
        Star.star_count += clock.tick(60)        # star_count gets updated with the time elapsed since the last frame (16ms)
                                                 # framerate is 60fps (aka we see 60 frames-images per sec), so 1 frame takes 16ms. The framerate depends on the number of sprites we have
        elapsed_time = time.time() - start_time

        if Star.star_count > time_interval: # When 2000ms have passed since last addition of stars, add 3 stars
            for _ in range(3):
                star = Star(random.randint(0, WIDTH - 10), -20, 10, 20)    # random x position for the star, y position is just above the window (minus sign) so that it doesn't spawn suddenly in the window

            time_interval = max(200, time_interval - 50)  # Decrease the time interval by 50ms (but keep it above 200ms). This way the game gets harder as time goes on
            Star.star_count = 0             # Reset the star_count

        my_player.move()

        for star in Star.stars[:]:       # Iterate over a copy of the list to avoid modifying the list while iterating (which can cause bugs)
            star.move()
            hit = star.check_collision(hit)
            if hit:
                break
        
        draw(elapsed_time)     # Update the window for this frame

        for event in pygame.event.get():      # pygame.event.get() returns a list of all the events that have occurred since the last frame
            if event.type == pygame.QUIT:     # If the user closes the window
                run = False                   # Exit the while loop in the next iteration
                break                         # Exit the for loop, there is no need to check for other events
        if hit:
            hit_draw()
            break                   # Exit the while loop

    pygame.quit()                   # Close the window

if __name__ == "__main__":      # This is the entry point of the program
    main()