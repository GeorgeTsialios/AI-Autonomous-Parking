## Καλές πρακτικές {#sec:training:tips}

Σε αυτήν την ενότητα, παρουσιάζονται κάποιες καλές πρακτικές στις εκπαιδεύσεις ενισχυτικής μάθησης, οι οποίες αξιοποιήθηκαν στην παρούσα εργασία και αποδείχθηκαν χρήσιμες για τη βελτίωση της απόδοσης των πρακτόρων. Όπως και πριν, θα εξετάσουμε τις συμβουλές αυτές πρώτα σε θεωρητικό επίπεδο, όπως περιγράφονται στη βιβλιογραφία, ενώ στη συνέχεια θα δούμε την εφαρμογή τους, στο περιβάλλον αυτόματης στάθμευσης.

### Επιλογή αλγορίθμου {#sec:training:tips:algorithms}

Στη βιβλιογραφία αναφέρεται πως δεν υπάρχει συγκεκριμένος κανόνας για το ποιος αλγόριθμος είναι καλύτερος, καθώς η απόδοση τους εξαρτάται από πολλούς παράγοντες. Επομένως, δεν υπάρχει κάποια φόρμουλα για την επιλογή αλγορίθμου, αλλά πρέπει να ληφθεί υπόψη το είδος του προβλήματος. Σημαντική είναι η διάκριση των χώρων καταστάσεων και ενεργειών του περιβάλλοντος, σε διακριτούς ή συνεχείς. Πιο αναλυτικά, υπάρχουν αλγόριθμοι που είναι ανεξάρτητοι αυτών των παραγόντων, όμως υπάρχουν κάποιοι που είναι κατάλληλλοι μόνο για συγκεκριμένες περιπτώσεις. Οι χώροι για τους οποίους είναι κατάλληλοι, οι αλγόριθμοι που χρησιμοποιήθηκαν στην παρούσα εργασία, παρουσιάζονται στον *Πίνακα @tbl:training:algorithms*:

| Αλγόριθμος | Χώρος Καταστάσεων (State Space) | Χώρος Ενεργειών (Action Space) | 
| ---- | -------- | ------- |
| Q-Learning | Διακριτός | Διακριτός |
| PPO | Συνεχής | Διακριτός ή Συνεχής |
| SAC | Συνεχής | Συνεχής |
| DDPG | Συνεχής | Συνεχής |
| TD3 | Συνεχής | Συνεχής |

Table: Πίνακας καταλληλότητας αλγορίθμων ενισχυτικής μάθησης {#tbl:training:algorithms}

Όπως φαίνεται από τον παραπάνω πίνακα, η επιλογή αλγορίθμου ήταν ένα πρώτο λάθος που κάναμε, ως αρχάριοι, σε αυτήν την εργασία. Συγκεκριμένα, η επιλογή του αλγορίθμου Q-Learning για την επίλυση του προβλήματος αυτόματης στάθμευσης ήταν άστοχη, καθώς ο αλγόριθμος δεν ενδείκνυται για περιβάλλοντα με συνεχή χώρο καταστάσεων, όπως το δικό μας. Έτσι, οι όποιες προσπάθειες έγιναν για την αντιμετώπιση αυτού του προβλήματος, όπως για παράδειγμα η διακριτοποίηση των καταστάσεων, δεν έφεραν τα επιθυμητά αποτελέσματα και σπαταλήθηκε άσκοπα, πολύτιμος χρόνος εκπαιδεύσεων. Επομένως, θα συμβούλευα τους νέους ερευνητές του πεδίου, να εξετάζουν προσεκτικά το state και το action space ενός αλγορίθμου, προτού προχωρήσουν σε εκπαιδεύσεις με αυτόν. 

### Επιλογή πολιτικής αλγορίθμου {#sec:training:tips:policy}

Κάποιοι αλγόριθμοι ενισχυτικής μάθησης, όπως ο DDPG και ο TD3, χρησιμοποιούν ντετερμινιστική πολιτική, ενώ άλλοι, όπως ο PPO και ο SAC, χρησιμοποιούν στοχαστική πολιτική. Η στοχαστική πολιτική, μπορεί να είναι χρήσιμη κατά την εκπαίδευση, ενθαρρύνοντας την εξερεύνηση του περιβάλλοντος, όμως, κατά την αξιολόγηση του πράκτορα, είναι καλό να θέτουμε πάντα την πολιτική του πράκτορα σε ντετερμινιστική. Με αυτόν τον τρόπο, είμαστε σε θέση να κρίνουμε καλύτερα τη συμπεριφορά που έχει μάθει ο πράκτορας, ενώ συνήθως βελτιώνονται και οι επιδόσεις του, αφού επιλέγονται πάντα οι ενέργειες, που έκρινε ως βέλτιστες, ο αλγόριθμος.

### Μοντελοποίηση προβλήματος {#sec:training:tips:problem}

#### Χώροι καταστάσεων και ενεργειών {.unnumbered}

Δύο σημαντικές παράμετροι της μοντελοποίησης του προβλήματος είναι η επιλογή της κατάστασης του περιβάλλοντος (είσοδος του πράκτορα) και των δυνατών ενεργειών του πράκτορα (έξοδος του πράκτορα). Συγκεκριμένα, είναι σημαντικό η κατάσταση του περιβάλλοντος να περιέχει μόνο χρήσιμες πληροφορίες για τον πράκτορα, προκειμένου να ελαχιστοποιηθεί το πλήθος των εισόδων του νευρωνικού δικτύου. Ακόμα, πρέπει ο πράκτορας να μπορεί να συσχετίσει τις ανταμοιβές που παίρνει, σε συγκεκριμένες καταστάσεις. Για παράδειγμα, η τιμωρία της σύγκρουσης μπορεί να γίνει κατανόητη από τον πράκτορα, καθώς παρατηρεί ότι τιμωρείται όταν η απόσταση ενός αισθητήρα γίνει 0. Μάλιστα, παρέχοντας ως είσοδο στον πράκτορα και την ταχύτητα του, θα είναι σε θέση να καταλάβει πως όταν η τιμή του αισθητήρα είναι μικρή και η ταχύτητα του μεγάλη, έπεται σύγκρουση. 

Αντίστοιχα, οι ενέργειες του πράκτορα πρέπει να επαρκούν για την επίλυση του προβλήματος. Μία σημαντική επιλογή που πρέπει να γίνει, είναι αυτή μεταξύ διακριτού ή συνεχούς χώρου ενεργειών. Ο διακριτός χώρος ενεργειών θεωρείται πως προσφέρει ταχύτερη και ευκολότερη εκπαίδευση, ενώ ο συνεχής, καλύτερες τελικές επιδόσεις. Στην περίπτωση μας, επιλέχθηκε διακριτός χώρος ενεργειών, ώστε ο πράκτορας να χειρίζεται το αυτοκίνητο με τα βέλη του πληκτρολογίου, όπως θα έκανε ένας άνθρωπος, προκειμένου να παίξει το παιχνίδι. Έτσι, ακόμα και όταν χρησιμοποιήθηκαν αλγόριθμοι με συνεχή χώρο ενεργειών, οι ενέργειες τους μετατράπηκαν στη συνέχεια σε διακριτές, ώστε να αντιστοιχούν στα βέλη του πληκτρολογίου. Με τον τρόπο αυτό, θεωρήσαμε πως μπορούμε να αξιολογήσουμε τους αλγορίθμους σε ίσους όρους, καθώς και να τους συγκρίνουμε με την ανθρώπινη επίδοση στο παιχνίδι. Μάλιστα, αυτό επιβεβαιώνεται, καθώς όταν υπό αυτές τις συνθήκες, συγκρίθηκε η χρήση διακριτών και συνεχών ενεργειών στον αλγόριθμο PPO, δεν παρατηρήθηκε καμία διαφορά στην επίδοση του πράκτορα.

#### Αραιές ανταμοιβές έναντι Διαμόρφωσης ανταμοιβής {.unnumbered}

Από την εμπειρία μας, η συνάρτηση ανταμοιβής ήταν ο παράγοντας που επηρέαζε στο μεγαλύτερο βαθμό, τις επιδόσεις των πρακτόρων. Για το πρόβλημα της αυτόματης στάθμευσης, δοκιμάστηκαν διαφορετικές συναρτήσεις ανταμοιβής, με πολλούς συνδυασμούς τιμών των ανταμοιβών τους και τα συμπεράσματα που προέκυψαν περιγράφονται παρακάτω.

Στη βιβλιογραφία, αναφέρεται πως μέσω αραιών ανταμοιβών, η εκπαίδευση του πράκτορα γίνεται συχνά πιο δύσκολη, εξαιτίας της έλλειψης επαρκούς ανάδρασης. Με άλλα λόγια, ο πράκτορας δεν έχει τρόπο να καταλάβει πόσο κοντά βρίσκεται στον τελικό στόχο του και σε κάποια περιβάλλοντα, είναι δύσκολο να επιλέξει τη σωστή ακολουθία ενεργείων, που θα τον οδηγήσει στο στόχο και στην ανταμοιβή του, χωρίς καμία παρότρυνση από τον σχεδιαστή. Για αυτό, η διαμόρφωση ανταμοιβής θεωρείται πιο εύκολη για τη μάθηση του πράκτορα και προτείνεται οι εκπαιδεύσεις να ξεκινάνε με αυτή τη μέθοδο [@youtubeAntonin].

Παρόλα αυτά, η διαμόρφωση ανταμοιβής έχει τις δικές της προκλήσεις. Συγκεκριμένα, οι τιμές των ανταμοιβών πρέπει να οριστούν πολύ προσεκτικά, ώστε να μην εμφανίζεται το φαινόμενο του reward hacking. Αντίστοιχα, μέσω των τιμών των ανταμοιβών πρέπει να γίνεται σαφής διαχωρισμός μεταξύ πρωτεύοντος στόχου (π.χ. στάθμευση) και δευτερευόντων στόχων (π.χ. αποφυγή σύγκρουσης), προκειμένου ο πράκτορας να επικεντρωθεί στον κύριο στόχο του και να μην αναπτύξει υποβέλτιστες πολιτικές. Γενικά, είναι σημαντικό ο σχεδιαστής να σκέπτεται από τη μεριά του πράκτορα, ο οποίος δεν γνωρίζει ποιός είναι ο στόχος του και προσπαθεί απλώς να μεγιστοποιήσει την ανταμοιβή του.

Τα παραπάνω προβλήματα της διαμόρφωσης ανταμοιβής είναι γνωστά στη βιβλιογραφία και αποδείχθηκαν στην περίπτωση μας, δύσκολο να ξεπεραστούν. Ακόμα, η μέθοδος των αραιών ανταμοιβών στάθηκε ικανή να επιτύχει την επιθυμητή απόδοση των πρακτόρων, σε ορισμένους αλγορίθμους. Για αυτούς τους λόγους, θα προέτρεπα μελλοντικούς ερευνητές να ξεκινήσουν από την σαφώς ευκολότερη μέθοδο των αραιών ανταμοιβών και εφόσον αυτή αποτύχει, να προχωρήσουν στη διαμόρφωση ανταμοιβής.

#### Συνθήκες τερματισμού επεισοδίου {.unnumbered}

Είναι συνηθισμένη πρακτική να ορίζεται ένας μέγιστος αριθμός βημάτων, ως συνθήκη τερματισμού του επεισοδίου και έναρξης του επόμενου, κατά την εκπαίδευση ενός πράκτορα ενισχυτικής μάθησης. Η τεχνική αυτή ονομάζεται *episode cutoffs* ή *timeouts*. Με τον τρόπο αυτό, αποφέυγονται τα ατέρμονα επεισόδια, στα οποία ο πράκτορας έχει κολλήσει σε μία μη τερματική κατάσταση και δεν μπορεί να προχωρήσει περαιτέρω. Επιπλέον, η μέθοδος αυτή, ωθεί τον πράκτορα να ανακαλύπτει πιο αποδοτικές πολιτικές, καθώς πρέπει να επιλύσει το πρόβλημα εντός συγκεκριμένου αριθμού βημάτων. Επομένως, ενθαρρύνεται η ταχύτητα και η αποτελεσματικότητα του πράκτορα.

Στην περίπτωση μας, επιλέξαμε το άνω όριο των 600 βημάτων για τα επεισόδια της εκπαίδευσης. Αυτό το όριο επιλέχθηκε, καθώς θεωρήθηκε πως 30 δευτερόλεπτα αρκούν για να λυθεί το πρόβλημα της αυτόματης στάθμευσης, κάτω από οποιεσδήποτε αρχικές συνθήκες ($30$ seconds $\times 20$ frames per second = $600$ frames ή steps). Ταυτόχρονα, τα 30 δευτερόλεπτα είναι ένα σχετικά μικρό χρονικό διάστημα, έτσι ώστε να ασκείται πίεση στον πράκτορα, να επιτύχει το στόχο του, εντός αυτού του χρόνου. Βέβαια, αξίζει να σημειωθεί, πως όταν δοκιμάσαμε να αφαιρέσουμε το όριο των 600 βημάτων, δεν παρατηρήσαμε μεγάλη διαφορά στα αποτελέσματα της εκπαίδευσης.

Μία άλλη τεχνική που εξετάστηκε, είναι αυτή του πρόωρου τερματισμού (*early stopping*). Η τεχνική αυτή, εισάγει επιπλέον συνθήκες για τον τερματισμό ενός επεισοδίου, όταν η απόδοση του πράκτορα σε αυτό, είναι πολύ μικρή. Για παράδειγμα, εφαρμόσαμε early stopping στα επεισόδια εκπαίδευσης, στην 3^η^ σύγκρουση του πράκτορα με κάποιο αντικείμενο. Το σκεπτικό πίσω από αυτήν την τεχνική, είναι πως αποτρέπει την εξερεύνηση άχρηστων συμπεριφορών από τον πράκτορα (όπως το να συγκρούεται συνεχώς με αντικείμενα) και με αυτόν τον τρόπο, μπορεί να επιταχύνει την εκπαίδευση. Χρειάζεται όμως ξανά, προσεκτικός σχεδιασμός, καθώς μπορεί να προκύψει κι έτσι, reward hacking. Στο προηγούμενο παράδειγμα, αν ο πράκτορας τιμωρείται με -10 για κάθε σύγκρουση αλλά και -1 για κάθε βήμα που περνάει, τότε μπορεί να υιοθετήσει την πολιτική του να συγκρούεται γρήγορα 3 φορές σε κάθε επεισόδιο, ώστε να τερματίζει το επεισόδιο με συνολική ανταμοιβή ~ -30, αντί για ~ -600, εφόσον ολοκληρωθεί το επεισόδιο χωρίς να παρκάρει. Ωστόσο, όπως και πριν, αυτή η τεχνική δεν προσέφερε στην πράξη, κάποια βελτίωση στην εκπαίδευση.

### Παράμετροι Εκπαίδευσης {#sec:training:tips:hyperparameters}

Στη βιβλιογραφία, αναφέρεται συχνά η επιρροή των παραμέτρων της εκπαίδευσης στην τελική επιτυχία αυτής. Ωστόσο στην πράξη, δεν παρατηρήσαμε μεγαλές αλλαγές στις επιδόσεις των πρακτόρων, από τη μεταβολή υπερ-παραμέτρων του αλγορίθμου, όπως ο ρυθμός μάθησης ή ο συντελεστής εντροπίας. Παρόμοια ήταν τα αποτέλεσματα και από την αλλαγή της δομής του νευρωνικού δικτύου. Συγκεκριμένα, τα μικρά νευρωνικά δίκτυα (1-2 κρυφά επίπεδα των 32-64 νευρώνων) θεωρούνται επαρκή για απλές εργασίες, ενώ τα πιο μεγάλα και πολύπλοκα δίκτυα (3-4 κρυφά επίπεδα των 256-512 νευρώνων) κρίνονται απαραίτητα για πιο δύσκολες εργασίες, εισάγοντας όμως τον κίνδυνο της υπερπροσαρμογής. Έτσι, προτείνεται αρχικά η χρήση μικρών δικτύων και η αύξηση της πολυπλοκότητας τους, μόνο αν αποδειχθεί αναγκαίο. Όμως, παρά τις αλλαγές στο μέγεθος των δικτύων που δοκιμάσαμε, δεν πετυχάμε βελτίωση των επιδόσεων των πρακτόρων, με αυτόν τον τρόπο.

Βέβαια, οι παρατηρήσεις μας αυτές, μάλλον οφείλονται στη χρήση της βιβλιοθήκης Stable-Baselines3, η οποία παρέχει προεπιλεγμένες τιμές για τις υπερ-παραμέτρους των αλγορίθμων και την αρχιτεκτονική των νευρωνικών δικτύων. Επομένως, αυτές οι παραμέτροι είναι επιλεγμένες με βάση τις τιμές τους στις αρχικές δημοσιεύσεις των αλγορίθμων, καθώς και την εμπειρία των δημιουργών της βιβλιοθήκης. Έτσι, θεωρούνται σε μεγάλο βαθμό βελτιστοποιημένες και προτείνονται από τη βιβλιοθήκη, για τη χρήση των αλγορίθμων της [@youtubeAntonin]. Για αυτό, θα προτείναμε σε μελλοντικές εργασίες που χρησιμοποιούν αξιόπιστες υλοποιήσεις των αλγορίθμων, να επικεντρωθούν στη σωστή μοντελοποίηση του προβλήματος, όπως αυτή εξετάζεται στην Ενότητα @sec:training:modeling, και όχι στις τροποποιήσεις των υπερ-παραμέτρων.

### Κανονικοποιήσεις τιμών {#sec:training:tips:normalization}

Η τεχνική της κανονικοποίησης των τιμών, διαφόρων παραμέτρων της εκπαίδευσης, θεωρείται πως μπορεί να συμβάλει στην αύξηση της απόδοσης των πρακτόρων. 

#### Κανονικοποίηση καταστάσεων {.unnumbered}

Το διάνυσμα κατάστασης αποτελεί την είσοδο του νευρωνικού δικτύου του πράκτορα. Γενικά, τα νευρωνικά δίκτυα λειτουργούν καλύτερα, όταν όλες οι είσοδοι κάθε επίπεδου είναι μικρότερες ή ίσες της μονάδας. Για τα ενδιάμεσα επίπεδα του δικτύου, αυτό εξασφαλίζεται από τις συναρτήσεις ενεργοποίησης. Ωστόσο, για το πρώτο επίπεδο, πρέπει να φροντίσουν οι σχεδιαστές για την κανονικοποίηση της εισόδου του δικτύου, διαιρώντας την κάθε τιμή εισόδου, με τη μέγιστη δυνατή τιμή που μπορεί να λάβει. Για αυτό, στην παρούσα εργασία, κανονικοποιήσαμε τις εισόδους στο διάστημα [0, 1], για τους αλγορίθμους που χρησιμοποιούν την συνάρτηση ενεργοποίησης ReLU και στο διάστημα [-1, 1], για τους αλγορίθμους που χρησιμοποιούν την συνάρτηση ενεργοποίησης Tanh, ώστε τα δεδομένα εισόδου της Tanh να έχουν μέση τιμή 0.

#### Κανονικοποίηση ενεργειών {.unnumbered}

Όταν χρησιμοποιείται συνεχής χώρος ενεργειών (στην περίπτωση μας, στους αλγορίθμους SAC, DDPG, TD3), μία καλή πρακτική είναι η κανονικοποίηση του, ώστε να είναι συμμετρικός σε κάθε ενέργεια. Συνήθως, επιλέγεται η κανονικοποίηση των ενεργειών στο διάστημα [-1, 1], όπως έγινε και σε αυτήν την εργασία, καθώς οι περισσότεροι αλγόριθμοι ενισχυτικής μάθησης βασίζονται σε Γκαουσιανή κατανομή με μέση τιμή $μ=0$ και τυπική απόκλιση $σ=1$. Επομένως, η έλλειψη κανονικοποίησης του χώρου ενεργειών, μπορεί να βλάψει την εκπαίδευση και είναι δύσκολο να αποσφαλματωθεί [@sb3tips].

#### Κανονικοποίηση ανταμοιβών {.unnumbered}

Τέλος, συχνά στη βιβλιογραφία αναφέρεται και η τεχνική της κανονικοποίησης των ανταμοιβών σε κάποιο διάστημα, όπως π.χ. στο [0, 1]. Ωστόσο, αυτό δεν κρίνεται απαραίτητο, καθώς αυτό που θεωρείται σημαντικό, είναι η σχετική διαφορά μεταξύ των τιμών των ανταμοιβών. Συγκεκριμένα, ανταμοιβές που είναι πολύ μεγαλύτερες σε μέγεθος από τις υπόλοιπες, μπορεί να κυριαρχήσουν στην εκπαίδευση και ο πράκτορας να αφοσιωθεί σε αυτές. 

Ωστόσο, στο πρόβλημα της αυτόματης στάθμευσης, όταν η επιβράβευση για τη στάθμευση ήταν συγκρίσιμη με τις υπόλοιπες ανταμοιβές, τότε ο πράκτορας δεν επικεντρωνόταν σε αυτήν, αλλά ανέπτυσσε υποβέλτιστες πολιτικές. Επομένως, κρίναμε σκόπιμο, να αυξήσουμε σε μεγάλο βαθμό την ανταμοιβή της στάθμευσης, ώστε να γίνει σαφές στον πράκτορα, πως αυτή αποτελεί το βασικό στόχο του. Στη συνέχεια, μετά την επίτευξη αυτού του στόχου από τον πράκτορα, ενθαρρύναμε τη βελτίωση της πολιτικής του, για παράδειγμα αυξάνοντας την τιμωρία των συγκρούσεων. Άρα, η αρχική απόκλιση της ανταμοιβής της στάθμευσης σε σχέση με τις υπόλοιπες, κρίθηκε αναγκαία για την επιτυχία των εκπαιδεύσεων κι έτσι, δεν δοκιμάστηκε η κανονικοποίηση των ανταμοιβών.

### Παράκαμψη βημάτων {#sec:training:tips:frameskip}

Μία τεχνική η οποία συνηθίζεται στην εκπαίδευση πρακτόρων σε περιβάλλοντα παιχνιδιών, είναι το *FrameSkip*, δηλαδή η παράκαμψη βημάτων από τον πράκτορα. Συγκεκριμένα, η ενέργεια του πράκτορα επαναλαμβάνεται για έναν συγκεκριμένο αριθμό βημάτων, π.χ. για 4 βήματα. Οι ενημερώσεις του δικτύου του πράκτορα συμβαίνουν κανονικά σε κάθε βήμα, όμως μόνο ανά 4 βήματα, ανανεώνεται η ενέργεια του πράκτορα στο περιβάλλον. Με τον τρόπο αυτό, αποφεύγεται η υπερβολικά συχνή, εναλλαγή ενεργειών του πράκτορα (δηλαδή κάθε 1/20 του δευτερολέπτου στην περίπτωση μας) και επιταχύνεται η διαδικασία της εκπαίδευσης. Στην πράξη, διαπιστώσαμε πως η εφαρμογή της τεχνικής αυτής, εξομάλυνε την οδήγηση του πράκτορα και ήταν καθοριστική, για την τελική επιτυχία της εκπαίδευσης.

Η βιβλιοθήκη Stable-Baselines3 έχει υλοποιημένη μία κλάση για την τεχνική αυτή, η οποία ονομάζεται `MaxAndSkipEnv`. Ωστόσο, η κλάση αυτή, πέραν από τη λειτουργία του FrameSkip, εκτελεί επίσης και αυτή του Max-Pooling Over Frames, δηλαδή επιστρέφει τη μέγιστη τιμή του κάθε pixel, στα βήματα που παρακάμψαμε. Αυτό είναι κάτι επιθυμητό στην περίπτωση ενός περιβάλλοντος εκπαίδευσης Atari, αλλά όχι στο παιχνίδι της αυτόματης στάθμευσης, καθώς ο χώρος καταστάσεων μας δεν αποτελείται από την εικόνα του παιχνιδιού (*pixel data*). Για αυτό τροποποιήσαμε κατάλληλα τον κώδικα της κλάσης `MaxAndSkipEnv`, ώστε να εφαρμόζεται μόνο η τεχνική του FrameSkip.

### Επίπεδα δυσκολίας και Κλιμακωτή Μάθηση {#sec:training:tips:CL}

Συχνά, είναι δύσκολο να επιτύχει ο πράκτορας απευθείας την επιθυμητή συμπεριφορά, σε περιβάλλοντα με υψηλό βαθμό δυσκολίας. Για αυτό, προτείνεται η εκπαίδευση να ξεκινάει από απλοποιημένες εκδοχές του περιβάλλοντος (*toy problems*), με σκοπό να δείξει σε αυτές ο πράκτορας ορσιμένα σημάδια ζωής, δηλαδή κάποια πρώτα καλά αποτελέσματα, που αποδεικνύουν πως ο αλγόριθμος λειτουργεί σωστά [@JohnSchulman]. Στη συνέχεια, ο σχεδιαστής μπορεί να αυξήσει σταδιακά το βαθμό δυσκολίας του περιβάλλοντος, μέχρι να φτάσει στο επιθυμητό επίπεδο. 

Πράγματι, αυτή η προσέγγιση αποδείχθηκε πολύ χρήσιμη σε αυτήν την εργασία. Η δημιουργία των 4 επιπέδων δυσκολίας του παιχνίδιου (βλ. @sec:game:rules:difficulty), συνέβαλε σημαντικά στην ευκολότερη αποσφαλμάτωση της κάθε εκπαίδευσης. Έπειτα, αφού ο πράκτορας είχε φτάσει σε ικανοποιητικό επίπεδο επιδόσεων σε εύκολα επίπεδα, εφαρμόστηκε η τεχνική της Κλιμακωτής Μάθησης (*Curriculum Learning*) και ήταν καθοριστική για την επίτευξη της επιθυμητής συμπεριφοράς στα δυσκολότερα επίπεδα.

Συγκεκριμένα, η Κλιμακωτή Μάθηση είναι μία ειδική κατηγορία της Μεταφοράς Γνώσης (*Transfer Learning*), η οποία περιγράφει το πώς μπορεί η εμπειρία ενός πράκτορα σε μία εργασία μάθησης να τον βοηθήσει να μάθει καλύτερα, κάποια άλλη, σχετική εργασία [@Russell2021]. Για παράδειγμα, ένας πράκτορας ρομπότ που έχει μάθει να παίζει τέννις, θα είναι σε θέση να εκπαιδευτεί πιο εύκολα να παίζει ένα παρόμοιο παιχνίδι, όπως το πινγκ-πονγκ. Η Κλιμακωτή Μάθηση στηρίζεται σε αυτή την ιδέα και προτείνει την διαδοχική εκπαίδευση του πράκτορα, σε περιβάλλοντα αυξανόμενης δυσκολίας. Για παράδειγμα, στην περίπτωση μας, κανένας πράκτορας δεν κατάφερε να εκπαιδευτεί επιτυχώς, απευθείας στο δυσκολότερο επίπεδο του παιχνιδιού (επίπεδο 4), το οποίο εισάγει τη συνθήκη του χρονικού διαστήματος ακινησίας για τον καθορισμό της επιτυχούς στάθμευσης. Ωστόσο, οι πράκτορες που είχαν εκπαιδευτεί επιτυχώς στο επίπεδο 3 και είχαν μάθει να εισέρχονται εντός της θέσης στάθμευσης, με την επανεκπαίδευση τους στο επίπεδο 4, κατάφεραν να προσαρμόσουν τη γνώση τους, ώστε να μένουν πλέον ακίνητοι εντός της θέσης.

### Παρακολούθηση μετρικών {#sec:training:tips:metrics}

Η παρακολούθηση των γραφικών παραστάσεων, που σχεδιάζονται αυτόματα από το εργαλείο Tensorboard, είναι κρίσιμη για τη διαδικασία της εκπαίδευσης. Μέσω αυτών, λαμβάνεται η απόφαση για συνέχιση ή μη μίας εκπαίδευσης. Αυτό καταδεικνύεται συνήθως, από την τάση της γραφικής των ανταμοιβών, καθώς αυτή φανερώνει τις αλλαγές στην πολιτική του πράκτορα. Όταν η γραφική αυτή, έχει σχετικά σταθερή τιμή για ένα μεγάλο χρονικό διάστημα (π.χ. επί 1M steps), θεωρείται πως η εκπαίδευση έχει συγκλίνει και μπορεί να σταματήσει. Τότε, είναι ασφαλής η αξιολόγηση του πράκτορα, καθώς πρόκειται για την τελική επίδοση του [@empiricaldesignreinforcementlearning]. Μάλιστα, από την εμπειρία μας, παρατηρήσαμε πως οι περισσότεροι αλγόριθμοι συγκλίνουν στα πρώτα 2-4M steps εκπαίδευσης, ενώ μετά από αυτό το σημείο, η αύξηση των επιδόσεων είναι ελάχιστη.  

Επίσης, οι μετρικές του Tensorboard μπορεί να υποδείξουν τις τροποποίησεις, που πρέπει να γίνουν στην εκπαίδευση. Για παράδειγμα, όταν η καμπύλη των ανταμοιβών εμφανίζει μεγάλες διακυμάνσεις, τότε αυτό αποτελεί ένδειξη πως ο ρυθμός μάθησης είναι πολύ υψηλός και πρέπει να μειωθεί, προκειμένου να επιτευχθεί πιο σταθερή εκπαίδευση. Αντίστοιχα, στην περίπτωση όπου η καμπύλη των ανταμοιβών συγκλίνει πολύ νωρίς, τότε ίσως χρειάζεται η ρύθμιση της εντροπίας του αλγορίθμου (*entropy regularization*). Πιο αναλυτικά, η αύξηση του συντελεστή εντροπίας του αλγορίθμου ενθαρρύνει την εξερεύνηση του πράκτορα, καθώς ωθεί σε πιο ίση κατανομή των ενεργειών του. Ωστόσο, ένας υπερβολικά μεγάλος συντελεστής εντροπίας, θα προκαλέσει απλά την τυχαιότητα των ενεργειών του πράκτορα.

Τέλος, είναι προτιμότερο οι μετρικές να σχεδιάζονται με βάση τα βήματα εκπαίδευσης και όχι τα επεισόδια εκπαίδευσης. Με αυτόν τον τρόπο, εξασφαλίζεται πως διαφορετικές εκπαιδεύσεις συγκρίνονται στον ίδιο αριθμό δειγμάτων των πρακτόρων, ανεξαρτήτως των μηκών των επεισοδίων τους.

### Διατήρηση αρχείου εκπαιδεύσεων {#sec:training:tips:logging}

Μιά σημαντική διάσταση της διαδικασίας των εκπαιδεύσεων, είναι η συνεπής διατήρηση ενός αναλυτικού αρχείου. Η μέθοδος αυτή γίνεται απαραίτητη, όταν το πλήθος των εκπαιδεύσεων αυξάνεται και η διάρκεια τους μεγαλώνει [@amidLessonsLearned]. Τότε, ένα λεπτομερές αρχείο θα βοηθήσει τον σχεδιαστή να οργανώσει καλύτερα τις σκέψεις του και να μην ξεχνάει ποιές ιδέες έχουν δοκιμαστεί ήδη και ποιά ήταν τα αποτελέσματα τους.

Στο πρόβλημα της αυτόματης στάθμευσης, κρατήσαμε διαφορετικό αρχείο για κάθε αλγόριθμο που εξετάστηκε. Τα αρχεία αυτά βρίσκονται στο [αποθετήριο](https://github.com/GeorgeTsialios/Thesis) της εργασίας, στον φάκελο `parking_game/Saved-training/Όνομα-αλγορίθμου`. Σε κάθε φάκελο, βρίσκονται τα εξής στοιχεία:

- Υποφάκελοι για κάθε ξεχωριστή εκπαίδευση, οι οποίοι περιέχουν το αντίστοιχο αρχείο κώδικα, τις γραφικές παραστάσεις του Tensorboard και τα αρχεία .zip με τα βάρη των καλύτερων πρακτόρων
- Το αρχείο `Όνομα-Αλγορίθμου.txt`, το οποίο περιέχει τις εξής πληροφορίες για κάθε εκπαίδευση:
    - τι καινούργιο δοκιμάστηκε στη συγκεκριμένη εκπαίδευση και που αποσκοπεί
    - τα βήματα εκπαίδευσης
    - μία σύντομη, λεκτική περιγραφή της εικόνας των μετρικών
    - συμπεράσματα από την εξέταση του πράκτορα στο παιχνίδι

Η διαδικασία αυτή, αν και χρονοβόρα, αποδείχθηκε τελικά κρίσιμη, για την επιτυχία της εργασίας και για αυτό, την προτείνουμε ανεπιφύλακτα σε μελλοντικούς ερευνητές.