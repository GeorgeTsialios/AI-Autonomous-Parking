# Εκπαίδευση {#sec:training}

Στο Κεφάλαιο αυτό θα δουμε τις εκπαιδεύσεις από ΟΛΟΥΣ τους αλγορίθμους που μελετήσαμε θεωρητικά στο Κεφάλαιο 3.

## Μετατροπή σε Gym Environment

Open AI Gym which is a toolkit for developing and comparing reinforcement
learning algorithms 

## Προβλήματα

RL has a number of parameters
that can greatly affect how well the technique works, but there
is only limited guidance available for setting these parameters. Αναφέρεται στο Tuning Computer Gaming Agents using Q-Learning

Reinforcement learning is highly dependent on the quality of the reward function. If the reward function is poorly designed, the agent may not learn the desired behavior.

Reinforcement learning can be difficult to debug and interpret. It is not always clear why the agent is behaving in a certain way, which can make it difficult to diagnose and fix problems.

## Καλές Πρακτικές

κοιτα στο state και action space του αλγοριθμου

We want to supply the MLP with information about the game,
since the agent needs to LINK its feedback in the form of future
rewards to a situation in the game

Reinforcement learning technique
tend to produce curves with high fluctuations if learning rate
is high. Αναφέρεται στο Tuning Computer Gaming Agents using Q-Learning

Transfer -> Curriculum Learning

#### Επεισόδια εκπαίδευσης

Ένα επεισόδιο εκπαίδευσης (*episode*) αποτελείται από ένα σύνολο βημάτων εκπαίδευσης. 

Yes, it is common practice to set a maximum number of steps per episode when training a reinforcement learning (RL) agent. There are several reasons why this is done:

Preventing Infinite Loops: If an episode can continue indefinitely, the agent might get stuck in a loop or a state where it doesn't reach a terminal condition (like a win/loss or a goal state). Setting a maximum number of steps ensures that every episode eventually ends, even if the agent hasn't achieved the task.

Encouraging Efficiency: By limiting the number of steps, you encourage the agent to complete the task more efficiently. The agent learns that it needs to achieve its goal within a certain number of steps, which can lead to faster and more effective behavior.

## Q Learning

the size of the Q-table is 96.  Αναφέρεται στο Tuning Computer Gaming Agents using Q-Learning 
The Q-values are guaranteed to converge
by some schemas, such as exploring every (s, a)
Although the Qlearning theorem ensures convergence after every state/action
has been considered many (i.e. infinitely many) times, a
practical rule of thumb is to visit every state-action pair
about 10 times σύμφωνα με Lai
The main takeaway from this is that
the agent should have at least explored all possible actions and states before
stopping with randomness in order to have knowledge about what can be done
in each state. Σουηδοι

Πλήθος καταστάσεων
Note that this estimation gives a
very optimistic idea of the scale since it contains many game
states that are in practice not possible

#### Διακριτοποίηση

Συχνά, όταν το πεδίο τιμών των
ενεργειών είναι συνεχές, ακολουθείται μία διαδικασία διακριτοποίησης. Με αυτό τον
τρόπο, μειώνεται σε μεγάλο βαθμό η διαστασιμότητα του προβλήματος,
διευκολύνοντας τον πράκτορα να αποφασίσει τη βέλτιστη ενέργεια του σε κάθε
κατάσταση. Φυσικά είναι πιθανή η δημιουργία νέων προβλημάτων λόγω έλλειψης
ακρίβειας, όταν το συνεχές πεδίο μετατρέπεται σε μικρό πλήθος πεπερασμένων
ενεργειών. Επομένως, κατά τη διακριτοποίηση πρέπει να ληφθεί υπόψη η ζητούμενη
ακρίβεια.


ΣΟΥΗΔΟΙ
576 state-action pairs
The col and row variables are integer values
ranging from [-1, 0, 1] which states how the snake is positioned relative to
the food. Αυτό εκαναν για το φιδάκι οι Σουηδοι. Όμως αυτοί είχαν διακριτα τετραγωνακια, σε μας η διακριτοποιηση ισως οδηγει σε ελλειψη ακριβειας.

ϵ(t) = min + (max − min)e^(−dt)

Each benchmark consists
of letting the agent play 100 games in order to get a fair average

It is apparent that, the larger the grid size για το φιδακι, the slower it converges.  we see that training on a smaller
grid results in a faster training compared to a large one. This can be explained
by a larger size makes it more difficult for the agent to find food resulting in
a higher chance of dying along the way towards the food.
 Για αυτο κι εγω ξεκινησα με το level1, όπου ο πράκτορας κανει spawn κοντα στη θεση.

A positive reward encourages the agent to stay alive while a negative punishment forces the agent to
quickly hunt for the food. Για αυτό κι εγω τιμωρω τον πράκτορα όσο δεν παρκάρει

low decay rates converge significantly slower. the fastest rate of convergence for the orange line (d=0.1). If you use a smaller one (eg d=0.001), it takes a lot more episodes to actually converges to the same score as before. It is therefore not necessary to perform random actions after a while, in our case just after a couple of 100 episodes. 


a lower ϵmin leads to a faster convergence and a higher average score


#### Γιατί όχι DQN αλλά άλλοι αλγόριθμοι

Deep Q-networks (DQNs): A type of reinforcement learning algorithm that is based on deep learning. It is commonly used for tasks that involve complex environments and large state spaces.
The Atari games agents are trained using raw pixel input as a state representation. The drawback of using
raw pixel input data is that it requires enormous amounts
of computational power while an average personal computer
should be able to train an algorithm that makes use of higherorder inputs (a game-state feature space). Pac-Xon