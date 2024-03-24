import pygame
import time
import random
pygame.font.init()      # Initialize the font module, essential for rendering text on the screen

WIDTH, HEIGHT = 1000, 700                       # Set the width and height of the window
WIN = pygame.display.set_mode((WIDTH, HEIGHT))  # Create a window
pygame.display.set_caption("Space Dodge")       # Set the title of the window

BG = pygame.transform.scale(pygame.image.load("bg.jpeg"), (WIDTH, HEIGHT))  # Load the background image and scale it to the window size

PLAYER_WIDTH = 40
PLAYER_HEIGHT = 60

PLAYER_VEL = 5
STAR_WIDTH = 10
STAR_HEIGHT = 20
STAR_VEL = 3

# print(pygame.font.get_fonts())

FONT = pygame.font.SysFont("gadugi", 30)    # Select text font and size


def draw(player, elapsed_time, stars):      # updates the window, is called every frame
    WIN.blit(BG, (0, 0))                    # Draw the background image

    time_text = FONT.render(f"Time: {elapsed_time:.1f}s", 1, "white")   # 1 is the anti-aliasing level, keep it 1 for better quality
    WIN.blit(time_text, (10, 10))           # Draw the time text

    pygame.draw.rect(WIN, "red", player)   

    for star in stars:
        pygame.draw.rect(WIN, "white", star)

    pygame.display.update()                 # Essential for the changes to appear on screen

def hit_draw():                    # Draw function that is called when the player hits a star
    lost_text = FONT.render("You Lost!", 1, "white")
    WIN.blit(lost_text, (WIDTH/2 - lost_text.get_width()/2, HEIGHT/2 - lost_text.get_height()/2))   # Center the text
    pygame.display.update()
    pygame.time.delay(2000)

def main():
    run = True

    player = pygame.Rect(WIDTH/2 - PLAYER_WIDTH/2, HEIGHT - PLAYER_HEIGHT, PLAYER_WIDTH, PLAYER_HEIGHT)     # inintialize player position (x,y) and size (width, height)
    clock = pygame.time.Clock()     # Create a clock object to control the frame rate
    start_time = time.time()
    elapsed_time = 0

    time_interval = 2000       # Every time_interval ms (eg 2000ms), add 3 stars
    star_count = 0             # Counter to keep track of time (up to 2000ms)

    stars = []                 # Active stars
    hit = False

    while run:
        star_count += clock.tick(60)        # star_count gets updated with the time elapsed since the last frame eg. 16ms
        elapsed_time = time.time() - start_time

        if star_count > time_interval: # When 2000ms have passed since last addition of stars, add 3 stars
            for _ in range(3):
                star_x = random.randint(0, WIDTH - STAR_WIDTH)    # random x position for the star
                star = pygame.Rect(star_x, -STAR_HEIGHT, STAR_WIDTH, STAR_HEIGHT) # y position is just above the window (minus sign) so that it doesn't spawn suddenly in the window
                stars.append(star)

            time_interval = max(200, time_interval - 50)  # Decrease the time interval by 50ms (but keep it above 200ms). This way the game gets harder as time goes on
            star_count = 0             # Reset the star_count

        keys = pygame.key.get_pressed()       # pygame.key.get_pressed() returns a dictionary with the state of all keys
        if keys[pygame.K_LEFT] and player.x - PLAYER_VEL >= 0:  # the 2nd condition ensures that the player doesn't go out of the window
            player.x -= PLAYER_VEL
        if keys[pygame.K_RIGHT] and player.x + PLAYER_VEL + player.width <= WIDTH: # the 2nd condition ensures that the player doesn't go out of the window
            player.x += PLAYER_VEL

        for star in stars[:]:       # Iterate over a copy of the list to avoid modifying the list while iterating (which can cause bugs)
            star.y += STAR_VEL
            if star.y > HEIGHT:     # If the top of the star goes below the window, remove it
                stars.remove(star)
            elif star.y + star.height >= player.y and star.colliderect(player): # 1st condition checks if the bottom of the star is below the player (only then can the player collide with the star and thus we have to check the 2nd condition)
                hit = True
                break               # Exit the for loop, there is no need to check the rest of the stars

        draw(player, elapsed_time, stars)     # Update the window for this frame

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
