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

For more details, you can:

- Read the full project report (in Greek): [Project Report (PDF)](https://github.com/GeorgeTsialios/AI-Autonomous-Parking/blob/main/Report.pdf)
- Watch the video presentation (in Greek): [YouTube Video](https://www.youtube.com/watch?v=0QqAaAosaes&t=546s&ab_channel=GeorgeTsialios)

<br>

## Table of Contents

