# AI Autonomous Parking

## Summary

This project served as the thesis for my studies at the Electrical and Computer Engineering Department of the University of Patras. It consists of a parking game written in Python (with the Pygame library), where the player is tasked with parking a car in a randomly assigned parking spot, using the arrow keys. AI agents were trained to play the game, using OpenAI's Gymnasium and Stable-Baselines3 libraries. The following algorithms -which belong to the **Machine Learning** subcategory called **Reinforcement Learning**- were used: **Q-learning**, Proximal Policy Optimization (**PPO**), Soft Actor Critic (**SAC**), Deep Deterministic Policy Gradient (**DDPG**) and Twin Delayed Deep Deterministic Policy Gradient (**TD3**).

The final results were very promising, as the agents' in-game performance closely approached human levels of play. For example, here you can see the TD3 agent's evolution during training:

https://github.com/user-attachments/assets/6439dd0e-7345-487b-b06a-ad9488d4a52b

Here, you can see a side-by-side comparison of the trained agents from different algorithms.

https://github.com/user-attachments/assets/e8a77114-3826-470a-aee3-211f5e31f1b7

Lastly, here you can see a side-by-side comparison between the best agent (TD3 algorithm) and a human player (myself). Can you guess who controls which car?

https://github.com/user-attachments/assets/1c11c800-7b9e-4557-acf9-ae4f6022e546

Finally, I analyzed and compared these algorithms based on training time and final performance. I used the following equation to evaluate each player's in-game performance: 

**score** = 70 × **success_rate** + max (20 − 5 × **mean_collisions**, 0) + 10 × max ((1 − **mean_time** − 10) / 10 , 0)

As we can see, I took into account different factors of the player's gameplay, to judge their ability to park the car into the desired parking spot. These factors are summed up into a single metric (**score**), which has a highest value of 100 (perfect player). The following graph shows the scores of the 4 best algorithms.

![1](https://github.com/user-attachments/assets/1e8c06ac-1c2b-4b62-94c2-2a946557af2c)

This graph is another proof that the agents have become pretty good at the game.

You can find much more details in the project's PDF report: [Thesis/output/thesis_master.pdf](https://github.com/GeorgeTsialios/AI-Autonomous-Parking/blob/main/Report.pdf) and watch a video presentation on [Youtube](https://www.youtube.com/watch?v=0QqAaAosaes&t=546s&ab_channel=GeorgeTsialios). They are both in the Greek language.

## Table of Contents

