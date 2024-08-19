
Deep Q-networks (DQNs): A type of reinforcement learning algorithm that is based on deep learning. It is commonly used for tasks that involve complex environments and large state spaces.

Ωστόσο, τα σύγχρονα προβλήματα εξαιτίας της
μεγάλης πολυπλοκότητάς τους, έχουν θεσπίσει ορισμένους περιορισμούς ως προς
την εφαρμογή των αλγορίθμων της ενισχυτικής μάθησης. Η λύση δίδεται μέσω της
μίμησης ενός βιολογικού μηχανισμού, των νευρωνικών δικτύων
ενσωμάτωσή τους στους αλγορίθμους
ενισχυτικής μάθησης. Δημιουργήθηκε με αυτόν τον τρόπο ένα νέο επιστημονικό πεδίο,
αυτό της βαθιάς ενισχυτικής μάθησης

Συχνά στους αλγορίθμους ενισχυτικής μάθησης θέτονται περιορισμοί λόγω της
διαστασιμότητας των προβλημάτων που καλούνται να αντιμετωπίσουν. Οι περιορισμοί
αυτοί οφείλονται κατά βάση στο μεγάλο πλήθος των διαφορετικών καταστάσεων και
δράσεων, όπου καθίσταται χρονοβόρα, κοστοβόρα και τελικά απαγορευτική η
ικανοποιητική εξερεύνηση του περιβάλλοντος. Παράλληλα, οι πολλά υποσχόμενες
δυνατότητες των τεχνητών νευρωνικών δικτύων στην επίλυση πολύπλοκων μη
γραμμικών μοντέλων, ώθησαν τους επιστήμονες του χώρου της ενισχυτικής μάθησης
σε μία προσπάθεια αξιοποίησής τους.

απλά προβλήματα μπορούν να λυθούν μέσω q learning,
όπου μετά από μερικές επαναλήψεις ο πίνακας κατάστασης-κίνησης
συμπληρώνεται και επομένως το πρόγραμμα γνωρίζει την βέλτιστη κίνηση σε κάθε
κατάσταση. Λύσεις τέτοιου είδους όμως δεν είναι εφικτές όταν η πολυπλοκότητα
του προβλήματος αυξάνεται. Όταν λοιπόν το πρόβλημα είναι επαρκώς μεγάλο το qtable που θα χρειαζόταν για να αποθηκευτούν όλες οι δυνατές καταστάσεις με τις
αντίστοιχες κινήσεις τους καταλήγει να είναι τεράστιο και μη βιώσιμο σε μέγεθος,
για τον λόγο αυτό χρησιμοποιούνται νευρωνικά δίκτυα στη θέση αυτού του q-table.

Πλέον, ο πράκτορας εκφράζεται ως ένα βαθύ νευρωνικό δίκτυο το οποίο
αναπροσαρμόζει τα βάρη του κατά τη διάρκεια της εκπαίδευσης, με στόχο την επίτευξη
της μέγιστης συνολικής ανταμοιβής. Αυτός ο τρόπος μάθησης αποτελεί και τον
πλησιέστερο του ανθρώπινου τρόπου μάθησης και σκέψης

Η είσοδος των δικτύων είναι η κατάσταση στην οποία βρίσκεται κάθε χρονική στιγμή ο
πράκτορας.

η έξοδος του χρησιμοποιούμενου νευρωνικού δικτύου αντιστοιχεί
είτε στις τιμές αξίας δράσης κατάστασης, δηλαδή στον πίνακα Q(s,a), είτε σε
πιθανότητες δράσης π(α|s), οπότε και γίνεται λόγος για δίκτυο αξίας και δίκτυο
πολιτικής αντίστοιχα

 The
number of layers and nodes in each layer are hyperparameters of the architecture.

learn these weights through several iterations of feed-forward and backward
propagation of training data through the network.

### Τεχνητά Νευρωνικά Δίκτυα

Βιολογικά Νευρωνικά Δίκτυα

All inputs are then multiplied by their respective weights and then
summed. Afterward, the output is passed through an activation function, which determines the
output. If that output exceeds a given threshold, it “fires” (or activates) the node, passing data to the
next layer in the network. This results in the output of one node becoming in the input of the next
node. This process of passing data from one layer to the next layer defines this neural network as a
feedforward network.

Larger weights signify that particular
variables are of greater importance to the decision or outcome.

Neural networks leverage sigmoid neurons, which are distinguished by having values between 0 and

 Η
συνήθης λειτουργία ενός νευρώνα είναι η εξής: ο νευρώνας λαμβάνει ένα σήμα, το
επεξεργάζεται και έπειτα το μεταδίδει στους νευρώνες με τους οποίους είναι
συνδεδεμένος.
Ο κάθε νευρώνας και οι συνδέσεις του έχουν κάποιο βάρος το οποίο μεταβάλλεται
κατά τη διάρκεια της εκπαίδευσης. Το βάρος αυτό δίνει προτεραιότητα σε κάποιους
νευρώνες και κάποιες συνδέσεις, ενώ αγνοεί άλλες. Ουσιαστικά αυτό είναι που
οδηγεί στην μάθηση του νευρωνικού, όσο μαθαίνει το νευρωνικό το βάρος αυτό
μεταβάλλεται αλλά όταν εκπαιδευτεί πλήρως τότε μπορεί να εμπιστευτεί ότι τα
σημαντικά σήματα θα μεταβιβαστούν λόγω τον βαρών και θα παρθεί η σωστή
απόφαση.
Το κάθε νευρωνικό δίκτυο αποτελείται από 3 βασικά επίπεδα. Το στρώμα εισόδου,
τα ενδιάμεσα στρώματα (τα οποία μπορεί να είναι 1 ή περισσότερα) και το στρώμα
εξόδου

Η πιο συνηθισμένη τοπολογία των νευρωνικών δικτύων είναι αυτή των πλήρως
συνδεδεμένων στρωμάτων. Όπως έχει ήδη αναφερθεί, η πλήρως συνδεδεμένη
τοπολογία συναντάται όταν ο κάθε νευρώνας ενός επιπέδου συνδέεται πλήρως με
όλους τους νευρώνες του επόμενου επιπέδου

Τα νευρωνικά δίκτυα όπως προαναφέρθηκε μαθαίνουν
ανακαλύπτοντας όλο και καλύτερα βάρη για να κάνουν καλύτερες
προβλέψεις, αυτό επιτυγχάνεται μέσω διάφορων αλγορίθμων
βελτιστοποίησης 

συνάρτηση ενεργοποίησης η οποία υπάρχει
για να αντιστοιχήσει τις εισόδους που μπορεί να είναι πολύ μεγάλες τιμές
στις αναμενόμενες τιμές εξόδου .

Στα βιολογικά νευρωνικά δίκτυα, η μετάδοση της πληροφορίας μπορεί να
αποτυπωθεί ως μια μη γραμμική συνάρτηση. Για να προσομοιωθεί αυτή η
συμπεριφορά παρεμβάλλουμε μια συνάρτηση ενεργοποίησης ανάμεσα στην έξοδο
του κάθε νευρώνα και της εισόδου του επόμενου στρώματος

Συνάρτηση υπερβολικής εφαπτομένης (tanh)
Συνάρτηση ενεργοποίησης ReLU
Η συνάρτηση ReLU έχει 2 σημαντικά προτερήματα: πρώτον, η υπολογιστική δύναμη
που απαιτεί είναι σχετικά μικρή, τουλάχιστον σε σύγκριση με τη σιγμοειδή και την
συνάρτηση υπερβολικής εφαπτομένης. Επίσης η ReLU είναι μη γραμμική αν και
φαίνεται πως είναι γραμμική. Στη πραγματικότητα η ReLU παρουσιάζει γραμμική
συμπεριφορά για τιμές μεγαλύτερες του μηδενός, ενώ μηδενίζει αρνητικές τιμές.
Αυτό όμως την οδηγεί στο μεγαλύτερο της μειονέκτημα το γνωστό και ως dying
ReLU problem, όπου αρνητικές τιμές οδηγούν στην ανικανότητα του δικτύου να
κάνει backpropagation και να μη μπορεί να μάθει

Αλγόριθμοι Βελτιστοποίησης

τα βάρη για τους
νευρώνες αρχικοποιούνται τυχαία. Τα βάρη αυτά προφανώς πρέπει να
τροποποιηθούν. Αυτό επιτυγχάνει η διαδικασία backpropagation. [16]

Γενικά, η διαδικασία εκπαίδευσης για ένα νευρωνικό δίκτυο συμβαίνει σε 5 στάδια:
• Αρχικοποίηση: Δίνονται τυχαίες τιμές στα βάρη του νευρωνικού.
• Forward propagation: Οι είσοδοι του νευρωνικού περνάνε μέσα από τους
νευρώνες και γίνεται ο υπολογισμός της εξόδου.
• Συνάρτηση σφάλματος: Εφαρμόζεται μια συνάρτηση σφάλματος για να
υπολογιστεί πόσο μακριά είναι το μοντέλο από το επιθυμητό
• Backpropagation: Τα αποτελέσματα της συνάρτησης σφάλματος
εξαπλώνονται από το τέλος προς την αρχή του δικτύου, αλλάζοντας στην
πορεία τους τα βάρη των νευρωνικών με στόχο την ελαχιστοποίηση της.
• Επανάληψη: η διαδικασία επαναλαμβάνεται με μικρές αλλαγές κάθε φορά.
Μετά από κάθε επανάληψη εφαρμόζεται ένας αλγόριθμος βελτιστοποίησης
που έχει σαν στόχο την ελαχιστοποίηση της συνάρτησης σφάλματος.

Ο αλγόριθμος Adam (Adaptive moment Estimation) λειτουργεί κάνοντας χρήση
ορμών πρώτης και δεύτερης τάξης[22] Το υπολογιστικό κόστος του Adam μπορεί να είναι υψηλό αλλά η μέθοδος είναι
εξαιρετικά γρήγορη και καταλήγει σε μικρό χρονικό διάστημα ενώ παράλληλα
διορθώνει το πρόβλημα της μείωσης του ρυθμού μάθησης και διατηρεί
ικανοποιητική διακύμανση για να μην παγιδευτεί σε τοπικό ελάχιστο.
Γι’ αυτούς τους λόγος στην εργασία αυτή έγινε επιλογή του αλγορίθμου Adam για
την βελτιστοποίηση


Υπερ-παράμερτοι: Πρόκειται για τις εξωτερικές παραμέτρους που ορίζονται από
τον χειριστή του νευρωνικού. Παράμετροι αυτού του είδους έχουν τεράστια
επίδραση στην αποτελεσματικότητα του δικτύου επομένως η επιλογή τους πρέπει
να γίνει μετά από προσεκτική σκέψη. Δεν υπάρχουν προκαθορισμένοι τρόποι για να
γίνει η επιλογή των υπερ-παραμέτρων, και επομένως η επιλογή τους πρέπει να
γίνει μέσω πειραματισμού.

οι βασικές παράμετροι αυτού του
είδους και οι συνήθεις τιμές τους:
- Ο αριθμός των κρυφών επιπέδων: Η αύξηση του αριθμού των κρυφών
επιπέδων μπορεί να βελτιώσει την ακρίβεια του νευρωνιού, όμως η
προσθήκη υπερβολικά πολλών επιπέδων από ένα σημείο και μετά απλώς
αυξάνει την πολυπλοκότητα χωρίς να προσφέρει ουσιαστικό όφελος
- Η συνάρτηση ενεργοποίησης:  η επιλογή της σωστής έχει σημαντική επίδραση στην ικανότητα του
δικτύου να μάθει και την ταχύτητα της μάθησης.
- O ρυθμός μάθησης: Ο ρυθμός μάθησης του νευρωνικού δικτύου είναι ίσως
η σημαντικότερη υπερ-παράμετρος.
- Μέγεθος τεμαχίων: Πολλές φορές μία εποχή πρέπει να χωριστεί σε
μικρότερα κομμάτια λόγω αυξημένης πολυπλοκότητας του δικτύου. Το
μέγεθος αυτών των τεμαχίων
Αλγόριθμος βελτιστοποίησης:


Το βασικό πλεονέκτημα των νευρωνικών δικτύων είναι ότι μπορούν να
χρησιμοποιηθούν ευέλικτα για όλα αυτά τα προβλήματα αρκεί να τροφοδοτηθούν
με αρκετά δεδομένα για να εκπαιδευτούν. Το γεγονός ότι έχουμε περάσει στην
εποχή της πληροφορίας μας επιτρέπει να έχουμε μια πληθώρα δεδομένων για
σχεδόν όλους τους τομείς της ανθρώπινης εμπειρίας γεγονός που έχει
δημιουργήσει ιδανικές συνθήκες για την εδραίωση των νευρωνικών δικτύων.
 
Types of NNs:

The perceptron is the oldest neural network, created by Frank Rosenblatt
Feedforward neural networks, or multi-layer perceptrons (MLPs), are comprised of an input
layer, a hidden layer or layers, and an output layer. While these neural networks are also
commonly referred to as MLPs
Convolutional neural networks (CNNs) are similar to feedforward networks, but they’re
usually utilized for image recognition, pattern recognition, and/or computer vision
Recurrent neural networks (RNNs) are identified by their feedback loops. These learning
algorithms are primarily leveraged when using time-series data to make predictions about
future outcomes,
προβλήματα όπου η πληροφορία στην είσοδο τους είναι
διαδοχικής φύσεως. Τα RNN καθώς δέχονται καινούργιες εισόδους κρατάνε στην
μνήμη τους τις παλαιότερες. Τα RNN λόγω των ιδιαιτεροτήτων τους χρησιμοποιούνται κατά βάση σε
προβλήματα που αφορούν κατανόηση συνεχόμενης πληροφορίας, όπως ομιλία.

### PPO

Καθώς γίνεται χρήση αλγορίθμου Proximal Policy Optimization, η
λογική που ακολουθείται είναι actor – critic κάνουμε χρήση δύο νευρωνικών
δικτύων, ένα που θα εκτελεί την λειτουργία του critic (στόχος αυτού του
δικτύου είναι η αξιολόγηση της κατάστασης στην οποία βρίσκεται ο
πράκτορας, και ένα που θα εκτελεί τη λειτουργία του actor (στόχος αυτού του
δικτύου είναι να διαλέγει την κίνηση βάση της τρέχουσας κατάστασης). Τα
δίκτυα αυτά θα έχουν ως είσοδο την κατάσταση του ταμπλό κάθε φορά που θα
χρησιμοποιούνται και ως έξοδο θα επιλέγουν μία από τις 312 δυνατές κινήσεις.
Γι’ αυτό το λόγο γίνεται χρήση δικτύων τριών επιπέδων, για τον actor γίνεται
χρήση ενός επιπέδου με είσοδο 100 νευρώνων και έξοδο 512 νευρώνων,
ακολουθεί tanh activation layer, έπειτα υπάρχει ένα fully connected στρώμα
512x512 νευρώνων ακολουθούμενο από tanh activation layer και τέλος ένα
512x312 fully connected στρώμα που δίνει ως έξοδο την softmax διανομή
πιθανοτήτων για την κάθε κίνηση βάση προτίμησης και γίνεται επιλογή μιας
κίνησης. Καθώς η κάθε επιλογή γίνεται βάση πιθανοτήτων, η «καλύτερη»
κίνηση θα γίνεται τις περισσότερες φορές αλλά όχι πάντα, γεγονός που
επιτρέπει την επιλογή διαφόρων κινήσεων και λύνει το exploration dilemma.
Εφαρμόζεται ένα clip loss function το οποίο δεν επιτρέπει στον πράκτορα να
αποκλίνει πάρα πολύ από την παλαιότερη πολιτική σε κάθε βήμα, λύνοντας
έτσι το πρόβλημα απόκλισης που αντιμετωπίζουν οι περισσότερες actor-critic
μέθοδοι.
Στο τέλος του κάθε επεισοδίου γίνεται το memory update όπου γίνεται η
εκτίμηση και η βελτιστοποίηση της πολιτικής βάση των καταστάσεων, των
κινήσεων και των επιβραβεύσεων που συναντήθηκαν στο επεισόδιο.


ΜΕΙΟΝΕΚΤΗΜΑΤΑ - ΔΥΣΚΟΛΙΕΣ RL

Reinforcement learning is highly dependent on the quality of the reward function. If the reward function is poorly designed, the agent may not learn the desired behavior.

Reinforcement learning can be difficult to debug and interpret. It is not always clear why the agent is behaving in a certain way, which can make it difficult to diagnose and fix problems.