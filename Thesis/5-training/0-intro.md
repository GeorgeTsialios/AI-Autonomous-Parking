# Εκπαίδευση {#sec:training}

#### Επεισόδια εκπαίδευσης

Ένα επεισόδιο εκπαίδευσης (*episode*) αποτελείται από ένα σύνολο βημάτων εκπαίδευσης. 

Yes, it is common practice to set a maximum number of steps per episode when training a reinforcement learning (RL) agent. There are several reasons why this is done:

Preventing Infinite Loops: If an episode can continue indefinitely, the agent might get stuck in a loop or a state where it doesn't reach a terminal condition (like a win/loss or a goal state). Setting a maximum number of steps ensures that every episode eventually ends, even if the agent hasn't achieved the task.

Encouraging Efficiency: By limiting the number of steps, you encourage the agent to complete the task more efficiently. The agent learns that it needs to achieve its goal within a certain number of steps, which can lead to faster and more effective behavior.


#### Διακριτοποίηση

Συχνά, όταν το πεδίο τιμών των
ενεργειών είναι συνεχές, ακολουθείται μία διαδικασία διακριτοποίησης. Με αυτό τον
τρόπο, μειώνεται σε μεγάλο βαθμό η διαστασιμότητα του προβλήματος,
διευκολύνοντας τον πράκτορα να αποφασίσει τη βέλτιστη ενέργεια του σε κάθε
κατάσταση. Φυσικά είναι πιθανή η δημιουργία νέων προβλημάτων λόγω έλλειψης
ακρίβειας, όταν το συνεχές πεδίο μετατρέπεται σε μικρό πλήθος πεπερασμένων
ενεργειών. Επομένως, κατά τη διακριτοποίηση πρέπει να ληφθεί υπόψη η ζητούμενη
ακρίβεια.
