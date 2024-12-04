# AI Autonomous Parking 

## Summary 

### Subject 

This project served as the thesis for my studies in the Electrical and Computer Engineering Department at the University of Patras. It features a parking game developed in **Python** using the **Pygame** library, where players are tasked with parking a car in a randomly assigned parking spot using arrow keys. AI agents were trained to play the game using **OpenAI's Gymnasium** and **Stable-Baselines3** libraries. The following algorithms -which fall under the **Reinforcement Learning** category of **Machine Learning**- were implemented:

- **Q-learning**
- Proximal Policy Optimization (**PPO**)
- Soft Actor Critic (**SAC**) 
- Deep Deterministic Policy Gradient (**DDPG**)
- Twin Delayed Deep Deterministic Policy Gradient (**TD3**)

<br>

### Visual Results

The results were very promising, with the agents' in-game performance closely approaching human levels of play. Below are some visual demonstrations:

<br>

- **TD3 Agent's Training Evolution:**

https://github.com/user-attachments/assets/6af80221-9004-4fb7-a67d-a4a98e767452

We can see that as the *Steps Trained* increase, the agent's performance improves.

<br>

- **Side-by-Side Comparison of Trained Agents Accross Algorithms:**

https://github.com/user-attachments/assets/d182e9b7-be53-4e21-ac07-7d9da5f042c9

We can see that all agents have developed strong parking skills.

<br>

- **Side-by-Side Comparison Between the Best Agent (TD3) and a Human Player (myself):**

https://github.com/user-attachments/assets/73fe3330-1704-40a0-8b96-a4b3ba9f3e28

*Can you guess who controls which car?*

<br>

### Performance Evaluation

To evaluate each player's performance, I devised the following scoring formula:

**score** = 70 × **success_rate** + max (20 − 5 × **mean_collisions**, 0) + 10 × max ((1 − **mean_time** − 10) / 10 , 0)

This formula considers various gameplay factors:

- **Success rate** – The ratio of successful parking attempts in 100 different scenarios.
- **Mean collisions** – The average number of collisions before parking.
- **Mean time** – The average time taken to park.

 These factors are combined in a single metric (**score**), which ranges from 0 to 100, with 100 representing a perfect player. Below is a graph showcasing the performance scores of the four best algorithms:
 
![1](https://github.com/user-attachments/assets/a362aba8-43be-4533-91df-3c5d594ef85b)

This graph highlights how effective the agents have become at playing the game.

<br>

### Additional Resources

Explore more about the project:

- [Project Report (PDF, in Greek)](https://github.com/GeorgeTsialios/AI-Autonomous-Parking/blob/main/Report.pdf)
- [Project Presentation (YouTube, in Greek)](https://www.youtube.com/watch?v=0QqAaAosaes&t=546s&ab_channel=GeorgeTsialios)
- [Best Agent Gameplay (8-minute YouTube video)](https://www.youtube.com/watch?v=dPb2kcen6cY)

<br>

## Help Guide

If you’d like to use this project, here’s a step-by-step guide to get you started.

### Play the Game

1. Navigate to [AI-Autonomous-Parking/parking_game/Saved-training/Human](https://github.com/GeorgeTsialios/AI-Autonomous-Parking/tree/main/parking_game/Saved-training/Human).
2. Choose the version of the game you want to play:
   - Run **instant.py** to play the *instant-parking* version of the game
   - Run **normal.py** file to play the *normal-parking* version of the game, where 2 seconds of immobility are required for the car to be considered parked.

<br>

### Watch the Trained Agents Play

For example, to watch the agent trained with the TD3 algorithm:

1. Navigate to [AI-Autonomous-Parking/parking_game/Saved-training/TD3/draw](https://github.com/GeorgeTsialios/AI-Autonomous-Parking/tree/main/parking_game/Saved-training/TD3/draw).
2. Choose the version of the game:
   - Run **TD3-instant.py** to watch the agent play the *instant-parking* version, 
   - Run **TD3-normal.py** to watch the agent play the *normal-parking* version.

<br>

### Build upon the project

#### Code and Training Results

1. Go to the **AI-Autonomous-Parking/parking_game/Saved-training/{algorithm_name}** folder to access the code and data for training agents with a specific algorithm.
2. Inside this folder, you’ll find:
   - Subfolders for each training attempt (e.g., 5B, 6A). Each subfolder contains:
     - A Python file (.py) with the code for that specific training attempt.
     - The best-trained agent for the attempt (.zip file).
   - A **Αρχείο {algorithm_name}.txt** file documenting the changes made and the results of each training attempt.
   - The **draw** subfolder, which contains the best-performing agent and its corresponding code.

#### Tensorboard logs

1. Navigate to **AI-Autonomous-Parking/parking_game/{algorithm_name}-logs**.
2. This folder contains subfolders with TensorBoard logs for each training attempt. These logs provide insights into the agent's progress during training.
