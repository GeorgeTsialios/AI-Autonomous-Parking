## Δυσκολίες Ενισχυτικής Μάθησης {#sec:training:difficulties}

Σε αυτήν την ενότητα θα αναλύσουμε τους διάφορους παράγοντες που καθιστούν τη διαδικασία της εκπαίδευσης πρακτόρων ενισχυτικής μάθησης, απαιτητική και πολύπλοκη. Θα εξετάσουμε τους λόγους δυσκολίας σε θεωρητικό επίπεδο, όπως αυτοί περιγράφονται στη βιβλιογραφία, ενώ παράλληλα θα αναφερθούμε και στην εμφάνισή τους, κατά τη διάρκεια των εκπαιδεύσεων που διεξήχθησαν. 

### Χρόνος εκπαίδευσης {#sec:training:difficulties:time}

Ίσως ο βασικότερος λόγος δυσκολίας της ενισχυτικής μάθησης, είναι η μη αποδοτικότητα των αλγορίθμων στη χρήση δειγμάτων από το περιβάλλον (*sample inefficient*), η οποία οδηγεί σε πολύ υψηλούς χρόνους εκπαίδευσης. Το άρθρο «Deep Reinforcement Learning Doesn't Work Yet» [@rlblogpost], το οποίο αναλύει τα κυριότερα προβλήματα της βαθιάς ενισχυτικής μάθησης, αναφέρει χαρακτηριστικά πως ο αλγόριθμος Rainbow DQN, ένας από τους πιο προηγμένους αλγορίθμους ενισχυτικής μάθησης, απαιτεί περίπου 83 ώρες εμπειρίας σε ένα απλό παιχνίδι Atari, προκειμένου να φτάσει τη μέση ανθρώπινη επίδοση. Ο χρόνος αυτός είναι πολύ μεγάλος, αν αναλογιστεί κανείς πως ένας άνθρωπος μπορεί να μάθει να παίζει το ίδιο παιχνίδι, σε μόλις μερικά λεπτά. 

Πράγματι, το πρόβλημα των μεγάλων χρόνων εκπαίδευσης επιβεβαιώθηκε στην εργασία μας, όπου συνήθως χρειαζόταν ένα χρονικό διάστημα περίπου ίσο με μία ημέρα, προκείμενου να μπορούμε να κρίνουμε με ασφάλεια την πορεία της εκπαίδευσης. Αυτό μετατρέπει τη διαδικασία της εκπαίδευσης σε έναν μαραθώνιο, όπου οι αλλαγές που γίνονται, απαιτούν πολύ χρόνο για να αξιολογηθούν. Από την άλλη, οι σχεδιαστές του συστήματος, προσπαθούμε να μη σπαταλάμε χρόνο σε ανώφελες εκπαιδεύσεις, γιατί γνωρίζουμε πως ο χρόνος που έχουμε στη διάθεση μας πρέπει να χρησιμοποιηθεί αποδοτικά. Έτσι, ο μεγάλος χρόνος εκπαίδευσης, μας κάνει συχνά διστακτικούς απέναντι στον πειραματισμό, ωθώντας μας σε πιο συντηρητικές και γνώριμες επιλογές, για τα αποτελέσματα των οποίων είμαστε πιο σίγουροι.

### Αποσφαλμάτωση {#sec:training:difficulties:debugging}

Η ενισχυτική μάθηση είναι δύσκολη στην αποσφαλμάτωση (*debugging*) και στην ερμηνεία των αποτελεσμάτων της. Πιο αναλυτικά, δεν είναι πάντα σαφές γιατί ο πράκτορας συμπεριφέρεται με συγκεκριμένο τρόπο, καθώς υπάρχει πληθώρα παραγόντων που μπορεί να δυσχεραίνει την απόδοση του. Για παράδειγμα, ο σχεδιαστής του συστήματος πρέπει να αναλογιστεί την αρχιτεκτονική των νευρωνικών δικτύων, τις υπερ-παραμέτρους του αλγορίθμου, την ποιότητα της συνάρτησης ανταμοιβής ή ακόμα και της κατάστασης που δέχεται ως είσοδο ο πράκτορας, καθώς και των διαθέσιμων ενέργειών του. Επομένως, καθίσταται δύσκολη η σωστή διάγνωση των προβλημάτων της εκπαίδευσης και η επίλυσή τους. Η χρονοβόρα διαδικασία της αποσφαλμάτωσης και των τροποποιήσεων της εκπαίδευσης αναφέρεται και στο [@amidLessonsLearned]. Μάλιστα, χαρακτηριστική είναι η *Εικόνα @fig:training:difficulties:expectations*, όπου ο συγγραφέας παρουσιάζει τη διαφορά μεταξύ των εκτιμήσεων του, για την κατανομή του χρόνου σε ένα έργο ενισχυτικής μάθησης και της πραγματικότητας.

![Προσδοκίες και πραγματικότητα στην εκπαίδευση πρακτόρων ενισχυτικής μάθησης [@amidLessonsLearned].](5-training/figures/Expectation-vs-reality.png){#fig:training:difficulties:expectations width=90%}

Παρατηρούμε πως ο χρόνος της διόρθωσης παραμέτρων της εκπαίδευσης αποδείχθηκε πολύ μεγαλύτερος από τον αναμενόμενο. Αυτό εναρμονίζεται και με τη δική μας εμπειρία. Παρόλο που δεν καταγράψαμε ακριβώς το χρόνο υλοποίησης του κώδικα, αυτός ήταν σίγουρα, σημαντικά μικρότερος του χρόνου που διήρκησαν οι εκπαιδεύσεις και οι τροποποιήσεις τους, προκειμένου να επιτευχθούν ικανοποιητικά αποτελέσματα.

### Υπερ-παραμέτροι {#sec:training:difficulties:hyperparameters}

Όπως περιγράφεται και στο [@TuningGamingAgents], η ενισχυτική μάθηση απαιτεί την επιλογή τιμών για πολλές υπερ-παραμέτρους, οι οποίες μπορεί να επηρεάσουν σημαντικά την απόδοση του αλγορίθμου. Ωστόσο, η βιβλιογραφία παρέχει περιορισμένες κατευθυντήριες οδηγίες για τη ρύθμιση αυτών των παραμέτρων. Η επιλογή των σωστών τιμών για τις υπερ-παραμέτρους, μπορεί να χρειαστεί πολλές δοκιμές, καθιστώντας τη διαδικασία εκπαίδευσης χρονοβόρα και απαιτητική. 

Η ευαισθησία των αλγορίθμων στις τιμές των παραμέτρων εντοπίστηκε και σε αυτήν την εργασία. Για παράδειγμα, στην *Εικόνα @fig:training:difficulties:sensitivity*, παρουσιάζονται δύο διαφορετικές εκπαιδεύσεις του ίδιου αλγορίθμου, όπου η μονή διαφορά είναι μία μικρή μεταβολή της τιμής του συντελεστή εντροπίας.

![Δύο εκπαιδεύσεις του αλγορίθμου PPO: η εκπαίδευση 47 με entropy coefficient = 0.01 και η εκπαίδευση 47B με entropy coefficient = 0.](5-training/figures/sensitivity.png){#fig:training:difficulties:sensitivity width=100%}

Παρατηρούμε πως η μικρή αυτή μεταβολή στην τιμή της υπερ-παραμέτρου, οδήγησε πράγματι, σε άλλαγη της απόδοσης του αλγορίθμου, αφού ήδη από τα 1.4M steps, η γραφική των episode rewards της εκπαίδευσης 47Β συγκλίνει σε αρνητική τιμή, ενώ αυτή της εκπαίδευσης 47 συνεχίζει να αυξάνεται. Ωστόσο, πρέπει να σημειωθεί, πως οι βελτιώσεις της απόδοσης που πετύχαμε κάνοντας τέτοιου είδους μεταβολές ήταν πάντοτε μικρές και δεν αποτέλεσαν την αιτία για την επίτευξη της επιθυμητής συμπεριφοράς του πράκτορα. Με άλλα λόγια, επιβεβαιώσαμε την επιρροή των υπερ-παραμέτρων στην εκπαίδευση, όμως αυτές ποτέ δεν ήταν η ειδοποιός διαφορά μεταξύ μίας αποτυχημένης και μίας επιτυχημένης εκπαίδευσης.

### Σχεδίαση συνάρτησης ανταμοιβής {#sec:training:difficulties:reward}

Η ενισχυτική μάθηση εξαρτάται σε μεγάλο βαθμό, από την ποιότητα της συνάρτησης ανταμοιβής. Αν αυτή δεν είναι άρτια σχεδιασμένη, ο πράκτορας μπορεί να μη μάθει τη βέλτιστη πολιτική. Η σωστή σχεδίαση είναι πιο δύσκολη από ό,τι φαίνεται, καθώς η συνάρτηση ανταμοιβής πρέπει να αντικατοπτρίζει με ακρίβεια τον πραγματικό στόχο του πράκτορα και να τον ωθεί αποκλειστικά στην επιθυμητή συμπεριφορά. Συχνά, οι αστοχίες στην εκπαίδευση οφείλονται στην παρερμηνεία της συνάρτησης ανταμοιβής από τον πράκτορα. Αυτό είναι ένα πλεονέκτημα των έτοιμων περιβαλλόντων εκπαίδευσης (π.χ. Atari), που τα καθιστά δημοφιλή, καθώς η συνάρτηση ανταμοιβής είναι απλώς το σκορ του παιχνιδιού και δεν χρειάζεται να οριστεί από τον σχεδιαστή του συστήματος.

#### Πειρατεία της ανταμοιβής {.unnumbered}

Ένα φαινόμενο που μπορεί να προκύψει από την ελαττωματική σχεδίαση της συνάρτησης ανταμοιβής, είναι η **πειρατεία της ανταμοιβής** (*reward hacking*). Ο όρος αυτός, περιγράφει έναν έξυπνο, αντισυμβατικό τρόπο που βρήκε ο πράκτορας για να μεγιστοποιεί την ανταμοιβή του, χωρίς επιτυγχάνει τον πραγματικό στόχο, που είχαν κατά νου οι σχεδιαστές του [@stampyWhatReward]. Επομένως, ο πράκτορας ανακαλύπτει κενά στο περιβάλλον του ή εκμεταλλέυεται αστοχίες του λογισμικού, με αποτέλεσμα να πετυχαίνει μεγαλύτερη συνολική ανταμοιβή, από αυτήν που θα πετυχαίνε ακολουθώντας την επιθυμητή συμπεριφορά. Ένα παράδειγμα reward hacking, παρουσιάζεται με χιουμοριστικό τρόπο, στην *Εικόνα @fig:training:difficulties:hacking-funny*.

![Παράδειγμα reward hacking [@youtubeAntonin].](5-training/figures/reward-hacking-funny.png){#fig:training:difficulties:hacking-funny width=65%}

Μία ενδιαφέρουσα περίπτωση reward hacking προέκυψε στις εκπαιδεύσεις μας, όταν προσπαθούσαμε μέσω της διαμόρφωσης της ανταμοιβής (*reward shaping*, βλ. @sec:theory:reinforcement_learning:concepts), να ωθήσουμε τον πράκτορα να παρκάρει, δηλαδή να παραμείνει ακίνητος εντός της θέσης στάθμευσης. Για να το πετύχουμε αυτό, στη συνάρτηση ανταμοιβής επιβραβεύαμε τον πράκτορα, σε κάθε βήμα που βρισκόταν εντός της θέσης στάθμευσης (+100). Μάλιστα, η επιβράβευση αυτή, ήταν αντιστρόφως ανάλογη της ταχύτητας του πράκτορα, ώστε να τον ενθαρρύνουμε να μην κινείται. Τέλος, όταν ο πράκτορας παρέμενε για 2 συνεχόμενα δευτερόλεπτα ακίνητος εντός της θέσης, τότε θεωρούσαμε πως πάρκαρε επιτυχώς και του δίναμε μεγάλη επιβράβευση (+5000). Ωστόσο, από τις γραφικές παραστάσεις του Tensorboard, οι οποίες φαίνονται στην *Εικόνα @fig:training:difficulties:hacking*, παρατηρήσαμε το παράδοξο, πως η μέση ανταμοιβή του πράκτορα αυξανόταν πολύ πέραν της ανταμοιβής για τη στάθμευση του, ενώ το ποσοστό επιτυχίας του παρέμενε μικρό (~25%).

![Περίπτωση reward hacking στο περιβάλλον αυτόματης στάθμευσης. Ο πράκτορας έμαθε να πετυχαίνει μεγαλύτερη ανταμοιβή, χωρίς να παρκάρει.](5-training/figures/reward-hacking.png){#fig:training:difficulties:hacking width=100%}

Εξετάζοντας τον πράκτορα, παρατηρήσαμε πως είχε μάθει να εισέρχεται αρχικά στη θέση στάθμευσης, αλλά στη συνέχεια να κινείται διαδοχικά εμπρός-πίσω, ώστε να λαμβάνει συνεχώς τις μικρές επιβραβεύσεις που του δίναμε, όσο βρίσκεται εντός της θέσης. Έτσι, δεν τον ενδιέφερε να παρκάρει, καθώς οι συνολικές ανταμοιβές που λάμβανε από αυτή τη συμπεριφορά (της τάξης των 10000), ήταν πολύ μεγαλύτερες, από αυτήν του παρκαρίσματος (5000). Η λύση σε αυτό το πρόβλημα, δόθηκε από την πιο προσεκτική ρύθμιση των τιμών των επιβραβεύσεων, σε κάθε περίπτωση.

#### Υιοθέτηση υποβέλτιστης πολιτικής {.unnnumbered}

Παρόλο που το reward hacking αποτελεί ένα από τα πιο ενδιαφέροντα φαινόμενα της ενισχυτικής μάθησης, τέτοιες περιπτώσεις είναι αρκετά σπάνιες. Το πολύ συνηθέστερο πρόβλημα, που αντιμετωπίζουν οι πράκτορες ενισχυτικής μάθησης, είναι η σύγκλιση σε τοπικό μέγιστο της συνάρτησης ανταμοιβής, δηλαδή την υιοθέτηση υποβέλτιστης πολιτικής από τον πράκτορα. Αυτό σημαίνει πως ο πράκτορας έχει μάθει κάποια χρήσιμη συμπεριφορά, η οποία ενισχύει τη συνολική ανταμοιβή του. Ωστόσο, η συμπεριφορά αυτή, είναι πολύ μακρία από την επιθυμητή από τους σχεδιαστές του συστήματος.

Από την εμπειρία μας, αυτό ήταν το πιο συχνό πρόβλημα που συναντήσαμε, στις διάφορες εκπαιδεύσεις πρακτόρων. Ένα παράδειγμα υιοθέτησης υποβέλτιστης πολιτικής, το οποίο συνέβη πολλές φορές και σε διαφορετικούς αλγορίθμους, περιγράφεται παρακάτω. Αρχικά, στη συνάρτηση ανταμοιβής, προσθέτουμε μία μικρή τιμωρία στον πράκτορα για κάθε βήμα που περνάει (-9), με το σκεπτικό να τον ωθήσουμε να παρκάρει το συντομότερο δυνατόν. Επίσης, προσθέτουμε μία μεγαλύτερη τιμωρία (-100), η οποία δίνεται στον πράκτορα, όταν αυτός συγκρούεται με άλλα αντικείμενα. Ακόμα, υπάρχουν οι κατάλληλες επιβραβέυσεις, για όταν ο πράκτορας πλησιάζει τη θέση στάθμευσης, όταν εισέλθει σε αυτήν και όταν παρκάρει. Ωστόσο, η γραφική παράσταση της μέσης ανταμοιβής του πράκτορα για αυτήν την εκπαίδευση, συγκλίνει γρήγορα στην τιμή -5400, όπως φαίνεται στην *Εικόνα @fig:training:difficulties:suboptimal*.

![Παράδειγμα σύγκλισης σε τοπικό μέγιστο.](5-training/figures/suboptimal.png){#fig:training:difficulties:suboptimal width=100%}

Εξετάζοντας στην πράξη τον πράκτορα, παρατηρήσαμε πως είχε μάθει απλώς να μένει ακίνητος σε κάθε επεισόδιο, για όλη τη διάρκεια του (εξού και $-9 \times 600$ steps $= -5400$ average episode reward). Αυτό που προφανώς συνέβη, ήταν πως στην αρχή της εκπαίδευσης, όπου οι ενέργειες του πράκτορα ήταν τυχαίες, αυτός αναπόφευκτα, είχε συχνές συγκρούσεις με άλλα αντικείμενα. Έτσι, η συνολική ανταμοιβή του ήταν πολύ αρνητική και ο πράκτορας έμαθε πως η καλύτερη στρατηγική, για να μεγιστοποιήσει την ανταμοιβή του, είναι να μένει ακίνητος. Πράγματι, όπως βλέπουμε από την *Εικόνα @fig:training:difficulties:suboptimal*, ο πράκτορας βέλτιωσε σημαντικά τη συνολική ανταμοιβή του, όμως αυτή είναι πολύ μικρότερη από την ανταμοιβή που θα έπαιρνε, αν πάρκαρε. Ωστόσο, ο πράκτορας πλέον, έχει συγκλίνει σε αυτήν την πολιτική και δεν εξερευνεί παραπάνω το περιβάλλον. 

Επιχειρήσαμε να αντιμετώπισουμε αυτό το πρόβλημα, προσθέτωντας μία τιμωρία στην ανταμοιβή του πράκτορα, για κάθε βήμα που παρέμενε ακίνητος, ώστε να τον ενθαρρύνουμε να εξερευνήσει περισσότερο και να βρει τη βέλτιστη πολιτική. Παρόλα αυτά, η απάντηση του πράκτορα ήταν να μάθει να κάνει το γύρο της πίστας, ασταμάτητα. Με τον τρόπο αυτό, απέφευγε τόσο την τιμωρία για τις συγκρούσεις, όσο και την τιμωρία για την ακινησία, αλλά και πάλι, δεν ανέπτυξε την επιθυμητή συμπεριφορά. Επομένως, γίνεται κατανοητό πως το μέγεθος της κάθε ανταμοιβής πρέπει να επιλεγεί πολύ προσεκτικά, ώστε να μην παρακινήσει τον πράκτορα σε κάποια υποβέλτιστη συμπεριφορά.

Η τάση αυτή των πρακτόρων ενισχυτικής μάθησης να συγκλίνουν σε υποβέλτιστες πολιτικές, αντί για τις επιθυμητές από τους σχεδιαστές τους, ώθησε τον [@rlblogpost], να τους παρομοιάσει με «τεμπέλικους δαίμονες, που προσπαθούν εσκεμμένα να παρερμηνεύσουν την ανταμοιβή και αναζητούν ενεργά το πιο εύκολο τοπικό μέγιστο». Μία παρόμοια σκέψη κάναμε και εμείς, βλέποντας συμπεριφορές όπως αυτή που περιγράψαμε πριν. Θεωρήσαμε δηλαδή, πως είναι παραγωγικό να φανταζόμαστε τους πράκτορες σαν μικρά παιδιά, τα οποία προσπαθούν επίτηδες, να αντιτίθενται στη συμπεριφορά που τους ζητάμε. 

Προφανώς, αυτές οι θεωρήσεις προέρχονται από την απογοήτευση των σχεδιαστών και έχουν κυρίως, ψυχαγωγική διάθεση. Το πραγματικό πρόβλημα, έγκειται στη δυσκολία εύρεσης ισορροπίας στο δίλημμα εξερεύνησης-εκμετάλλευσης. Συγκεκριμένα, όταν ο πράκτορας εξερευνεί πολύ, τα δείγματα που συλλέγει είναι ανεπαρκή για να μάθει τη βέλτιστη συμπεριφορά. Αντίθετα, όταν εκμεταλλεύεται πολύ, κινδυνεύει να συγκλίνει πρόωρα, σε συμπεριφορές που δεν είναι βέλτιστες.

Δυστυχώς, παρόλο που το πρόβλημα της εξερεύνησης είναι από τα παλαιότερα του πεδίου της ενισχυτικής μάθησης και έχουν προταθεί διάφορες ιδέες κατά καιρούς για την αντιμετώπιση του, καμία δεν εγγυάται σταθερά αποτελέσματα σε όλα τα περιβάλλοντα. Ο λόγος για αυτό, είναι το μέγεθος της δυσκολίας του προβλήματος. Σύμφωνα με τη Wikipedia, το πρόβλημα εντοπίστηκε αρχικά από ερευνητές των συμμάχων στον 2^ο^ παγκόσμιο πόλεμο, όμως αποδείχθηκε τόσο δυσεπίλυτο, που προτάθηκε η διαρροή του σε Γερμανούς επιστήμονες, ώστε να σπαταλήσουν και αυτοί τον χρόνο τους πάνω σε αυτό [@wikipediaMultiarmedBandit].

### Αστάθεια της εκπαίδευσης {#training:difficulties:instability}

Ένας παράγοντας που προκαλεί αστάθεια στις εκπαιδεύσεις ενισχυτικής μάθησης, είναι το γεγονός πως τα δεδομένα συλλέγονται από τον ίδιο τον πράκτορα, κατά την αλληλεπίδραση με το περιβάλλον του. Αυτό έρχεται σε αντίθεση με την πολύ πιο σταθερή, επιβλεπόμενη μάθηση, όπου υπάρχει ένα στατικό σύνολο δεδομένων, καθορισμένο πριν την εκπαίδευση. Έτσι, στην ενισχυτική μάθηση, τα δεδομένα εκπαίδευσης εξαρτώνται από την πολιτική του πράκτορα. Αυτή η εξάρτηση, μπορεί να οδηγήσει σε φαύλο κύκλο: αν ο πράκτορας συλλέγει κακής ποιότητας δεδομένα (π.χ. καταστάσεις χωρίς ανταμοιβές), τότε δεν θα ανακαλύψει κάποια χρήσιμη συμπεριφορά κι έτσι θα συνεχίσει να συλλέγει κακής ποιότητας δεδομένα κοκ.

Επομένως, προκύπτει πως δύο εκπαιδεύσεις ενισχυτικής μάθησης, με τα ίδια ακριβώς χαρακτηριστικά, μπορεί να έχουν διαφορετικά αποτελέσματα η μία με την άλλη. Συγκεκριμένα, σε μία εκπαίδευση όπου τυχαία, ανακαλύπτονται νωρίς καλά δείγματα, η απόδοση του πράκτορα θα βελτιωθεί. Αντίθετα, σε μία εκπαίδευση όπου δεν ανακαλύπτονται χρήσιμα δείγματα από νωρίς, ο πράκτορας μπορεί να συγκλίνει σε υποβέλτιστες λύσεις, καθώς βλέπει πως όσα δοκιμάζει, έχουν χειρότερα αποτελέσματα.

Άρα, πέραν των υπολοιπών προβλημάτων των εκπαιδεύσεων ενισχυτικής μάθησης, αυτές παρουσιάζουν και εξάρτηση από την τύχη, δηλαδή από τους ψευδο-τυχαίους αριθμούς που παράγονται από το περιβάλλον. Για αυτό, προτείνεται πάντα να εκτελούνται πολλαπλές εκπαιδεύσεις με τα ίδια χαρακτηριστικά, ώστε τα αποτελέσματα να θεωρούνται αξιόπιστα [@sb3tips].