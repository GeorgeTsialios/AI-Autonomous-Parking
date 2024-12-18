## Μηχανική Μάθηση {#sec:theory:machine_learning}

### Ορισμός {#sec:theory:machine_learning:definition}

Ο όρος «Μηχανική Μάθηση» επινοήθηκε από τον Arthur Samuel το 1959 και περιγράφηκε ως «η ικανότητα ενός υπολογιστή να μαθαίνει χωρίς να προγραμματιστεί ρητά» [@5389202]. Ο Tom Mitchell έδωσε το 1997 έναν διάσημο, πιο μαθηματικό ορισμό των αλγορίθμων μηχανικής μάθησης: «Ένα πρόγραμμα υπολογιστή λέγεται ότι μαθαίνει από την εμπειρία Ε ως προς μια κλάση εργασιών Τ και μέτρο απόδοσης Ρ, αν η απόδοσή του στις εργασίες της κλάσης Τ, όπως μετράται από το μέτρο απόδοσης Ρ, βελτιώνεται με την εμπειρία Ε» [@Mitchell1997]. 

Επομένως, γίνεται σαφές ότι η μηχανική μάθηση ασχολείται με την ανάπτυξη αλγορίθμων, οι οποίοι μαθαίνουν να εκτελούν εργασίες μέσα από δεδομένα ή από προηγούμενη εμπειρία τους και όχι μέσω συγκεκριμένων εντολών. Αυτό είναι πολύ σημαντικό, καθώς τα παραγόμενα μοντέλα μηχανικής μάθησης είναι σε θέση να γενικεύουν τις γνώσεις τους και να τις εφαρμόζουν σε νέα δεδομένα. Το χαρακτηριστικό τους αυτό, τα κάνει ιδιαίτερα χρήσιμα για την επίλυση προβλημάτων, τα οποία δεν μπορούν να επιλυθούν με την κλασική προγραμματιστική προσέγγιση.  Ωστόσο, η εκπαίδευση των μοντέλων μηχανικής μάθησης απαιτεί μεγάλο όγκο δεδομένων, καθώς και χρόνο και πόρους για την επεξεργασία τους, ενώ το τελικό αποτέλεσμα εξαρτάται σε μεγάλο βαθμό από την ποιότητα των δεδομένων. Συγκεκριμένα, τα μοντέλα μπορεί να αναπτύξουν προκαταλήψεις (*bias*), εφόσον αυτές υπάρχουν στα δεδομένα εκπαίδευσης τους. Για παράδειγμα, η μελέτη των [@theconversationAgeismSexism], ανακάλυψε κοινωνικές προκαταλήψεις στην εφαρμογή παραγωγής εικόνων Midjourney. Όταν ζητήθηκε η παραγωγή εικόνων ανθρώπων σε εξειδικευμένα επαγγέλματα, τα αποτελέσματα απεικόνιζαν πάντα άνδρες, ενισχύοντας τη φύλετική προκατάληψη του ρόλου των γυναικών στον χώρο εργασίας.

### Κατάταξη πεδίου {#sec:theory:machine_learning:hierarchy}

Σήμερα, πολλοί συγχέουν τους όρους «Τεχνητή Νοημοσύνη» και «Μηχανική Μάθηση». Οφείλει να γίνει κατανοητό, πως η μηχανική μάθηση αποτελεί ένα υποσύνολο της τεχνητής νοημοσύνης, που εστιάζει στη βελτίωση της απόδοσης ενός συστήματος με βάση την εμπειρία. Ορισμένα συστήματα τεχνητής νοημοσύνης χρησιμοποιούν μεθόδους μηχανικής μάθησης, ενώ άλλα όχι. Αυτό φαίνεται με παραστατικό τρόπο, στην *Εικόνα @fig:theory:machine_learning:MLSubset*. 

![Ιεραρχία πεδίων τεχνητής νοημοσύνης [@MLSubset].](3-theory/figures/MLSubset.png){#fig:theory:machine_learning:MLSubset width=70%}

Μάλιστα, η εικόνα @fig:theory:machine_learning:MLSubset παρουσιάζει και τη σχέση μεταξύ μηχανικής μάθησης και βαθιάς μάθησης, δείχνοντας πως η βαθιά μάθηση αποτελεί μια περαιτέρω εξειδίκευση της μηχανικής μάθησης. Αμφότερες οι τεχνικές αυτές, χρησιμοποιούν τεχνητά νευρωνικά δίκτυα για να «μάθουν» από τα δεδομένα. Όμως, η βαθιά μάθηση χρησιμοποιεί πιο πολύπλοκα, πολυεπίπεδα νευρωνικά δίκτυα, τα οποία απαιτούν μεγαλύτερο όγκο δεδομένων και πόρων για την εκπαίδευσή τους. 

### Κατηγορίες {#sec:theory:machine_learning:categories}

Η μηχανική μάθηση χωρίζεται σε τρεις κύριες κατηγορίες, ανάλογα με τον τρόπο εκπαίδευσης των μοντέλων: την επιβλεπόμενη μάθηση, την μη επιβλεπόμενη μάθηση και την ενισχυτική μάθηση. Οι τρεις αυτές κατηγορίες παρουσιάζονται στην *Εικόνα @fig:theory:machine_learning:MLtypes* και περιγράφονται παρακάτω.

![Κατηγορίες μηχανικής μάθησης [@em360techWhatMachine].](3-theory/figures/test.png){#fig:theory:machine_learning:MLtypes width=80%}

#### Επιβλεπόμενη Μάθηση {.unnumbered}

Η Επιβλεπόμενη Μάθηση (*Supervised Learning*) είναι ο πιο συνηθισμένος τύπος μηχανικής μάθησης. Πήρε το όνομα της, καθώς η εκπαίδευση γίνεται υπό επίβλεψη, δηλαδή το μοντέλο μαθαίνει μέσω παραδειγμάτων. Συγκεκριμένα, παρέχεται στο μοντέλο ένα σύνολο δεδομένων εκπαίδευσης με ετικέτες (*labeled data*). Αυτό σημαίνει ότι κάθε δεδομένo, αποτελείται από ένα ζεύγος εισόδου-επιθυμητής εξόδου. Το μοντέλο κατά την εκπαίδευση, εντοπίζει μοτίβα στα δεδομένα και προβλέπει για κάθε είσοδο, ποιά είναι η αντίστοιχη έξοδος. Όταν κάνει λάθος, το μοντέλο αναπροσαρμόζεται, μέχρι να μάθει να αντιστοιχίζει σωστά τις εισόδους στις αντίστοιχες εξόδους. Το ζητούμενο είναι, το τελικό μοντέλο να μπορεί να χρησιμοποιηθεί για να κάνει προβλέψεις σε νέα δεδομένα -δηλαδή δεδομένα που δεν υπήρχαν στο σύνολο εκπαίδευσης- και να πετυχαίνει σε αυτά μεγάλο ποσοστό επιτυχίας.

Κλασικό παράδειγμα επιβλεπόμενης μάθησης αποτελεί η αναγνώριση του φύλου από εικόνες. Στο σύνολο δεδομένων εκπαίδευσης, κάθε εικόνα έχει ετικέτα με το φύλο του ατόμου που απεικονίζεται. Το μοντέλο εκπαιδεύεται να αναγνωρίζει τα χαρακτηριστικά που διαφοροποιούν τα δύο φύλα και να κάνει προβλέψεις για το φύλο του ατόμου. Υπάρχει περίπτωση, το μοντέλο να κάνει σωστές προβλέψεις στις εικόνες που εκπαιδεύτηκε, αλλά όχι σε άγνωστες εικόνες. Τότε, το μοντέλο δεν έχει μάθει να αναγνωρίζει σωστά το φύλο, αλλά απλά έχει μάθει να απαντάει σωστά στα δεδομένα εκπαίδευσης του. Το φαινόμενο αυτό ονομάζεται υπερπροσαρμογή ή υπερεκπαίδευση (*overfitting*) του μοντέλου και προφανώς, αποτελεί ανεπιθύμητη συμπεριφορά.

Επίσης, αξίζει να σημειωθεί πως όταν η έξοδος είναι μία τιμή από ένα πεπερασμένο σύνολο τιμών (όπως π.χ. πριν, άντρας/γυναίκα), τότε το πρόβλημα μάθησης ονομάζεται Ταξινόμηση (*Classification*). Αντίθετα, όταν η έξοδος είναι μία συνεχής τιμή (π.χ. η αυριανή θερμοκρασία), τότε το πρόβλημα ονομάζεται Παλινδρόμηση (*Regression*). Συνήθεις αλγόριθμοι που χρησιμοποιούνται για προβλήματα ταξινόμησης είναι τα Δέντρα Απόφασης (*Decision Trees*) και οι Μηχανές Διανυσμάτων Υποστήριξης (*Support Vector Machines*). Αντίθετα, για προβλήματα παλινδρόμησης χρησιμοποιούνται αλγόριθμοι όπως η Γραμμική Παλινδρόμηση (*Linear Regression*) και η Πολυωνιμική Παλινδρόμηση (*Polynomial Regression*).

#### Μη Επιβλεπόμενη Μάθηση {.unnumbered}

Η μη επιβλεπόμενη μάθηση (*Unsupervised Learning*) αναφέρεται στις περιπτώσεις όπου τα δεδομένα δεν έχουν ετικέτες, δηλαδή το μοντέλο δεν γνωρίζει την επιθυμητή έξοδο για κάθε δεδομένο. Πλέον, στόχος είναι η αναγνώριση μοτίβων, δομών ή συσχετίσεων στα δεδομένα. Οι πιο συνηθισμένες εργασίες μη επιβλεπόμενης μάθησης είναι η ομαδοποίηση και η μείωση διαστάσεων. 

Η Ομαδοποίηση (*Clustering*) αφορά την οργάνωση των δεδομένων, χωρίζοντας τα σε ομάδες με παρόμοια χαρακτηριστικά. Ένα παράδειγμα αποτελεί η οργάνωση των πελατών μίας επιχείρησης σε ομάδες με βάση το ιστορικό αγορών τους, έτσι ώστε να εφαρμοστεί διαφορετική στρατηγική προώθησης για κάθε ομάδα. Παραδείγματα αλγορίθμων ομαδοποίησης είναι ο K-Means και η Ιεραρχική Ομαδοποίηση (*Hierarchical Clustering*).

Η Μείωση Διαστάσεων (*Dimensionality Reduction*) αφορά τη μείωση του αριθμού των χαρακτηριστικών που περιγράφουν τα δεδομένα, διατηρώντας όμως σε μεγάλο βαθμό την πληροφορία που περιέχουν. Αυτό είναι χρήσιμο για την απλοποίηση μοντέλων, την αφαίρεση θορύβου από τα δεδομένα και την οπτικοποίηση τους . Ένας από τους πιο διάσημους αλγορίθμους μείωσης διαστάσεων είναι ο PCA (*Principal Component Analysis*).

#### Ενισχυτική Μάθηση {.unnumbered}

Η ενισχυτική μάθηση (*Reinforcement Learning*) αποτελεί την κατηγορία της μηχανικής μάθησης που μιμείται πιο πιστά τον τρόπο μάθησης των ανθρώπων. Ένας πράκτορας ενισχυτικής μάθησης εκπαιδεύεται μέσω της αλληλεπίδρασης με το περιβάλλον του και επιλέγοντας δράσεις σε αυτό. Ο πράκτορας λαμβάνει θετική ανταμοιβή, όταν οι ενέργειες του οδηγούν σε επιθυμητά αποτελέσματα και αρνητική ανταμοιβή, όταν οδηγούν σε ανεπιθύμητα αποτελέσματα. Μέσω δοκιμών και λαθών, ο πράκτορας μαθαίνει να παίρνει αποφάσεις που μεγιστοποιούν τις ανταμοιβές του. Η ενισχυτική μάθηση χρησιμοποιείται σε εφαρμογές όπως η ρομποτική, τα παιχνίδια και η διαχείριση πόρων. Μερικοί δημοφιλείς αλγόριθμοι ενισχυτικής μάθησης είναι ο Q-Learning, ο PPO (*Proximal Policy Optimization*) και ο SAC (*Soft Actor-Critic*).

Η ενισχυτική μάθηση αποτελεί την κατηγορία των αλγορίθμων μηχανικής μάθησης που χρησιμοποιούνται στην παρούσα εργασία και για αυτό, αναλύεται σε μεγαλύτερο βάθος στην επόμενη ενότητα.