#  Αξιολόγηση αποτελεσμάτων {#sec:evaluation}

Σε αυτό το κεφάλαιο, θα παρούσιασουμε τα αποτελέσματα από τις εκπαιδεύσεις των αλγορίθμων, με πιο οργανωμένο τρόπο, προκειμένου να προβούμε στη σύγκριση τους. 

Αρχικά, ας ορίσουμε τη σημασία της σύγκρισης αλγορίθμων ενισχυτικής μάθησης. Προφανώς, το κάθε περιβάλλον εκπαίδευσης έχει τα δικά του χαρακτηριστικά και ιδιαιτερότητες. Επομένως, δεν μπορούμε να γενικεύσουμε τα συμπεράσματα της εργασίας μας, σε όλα τα διαφορετικά περιβάλλοντα. Ωστόσο, η σύγκριση των αλγορίθμων θα είναι έγκυρη, για περιβάλλοντα με παρόμοιες ιδιότητες με αυτό της αυτόματης στάθμευσης. Επομένως, όπως αναφέρεται και στο [@empiricaldesignreinforcementlearning], συγκρίνοντας αλγορίθμους ενισχυτικής μάθησης κάνουμε τον ισχυρισμό: «αν το πρόβλημά σας είναι παρόμοιο με το δικό μας, τότε αυτός είναι ο αλγόριθμος που πρέπει να χρησιμοποιήσετε».

Ακόμα, μία καλή πρακτική κατά την εξέταση των επιδόσεων των πρακτόρων, είναι η ποσοτική αξιολόγηση [@youtubeAntonin]. Συγκεκριμένα, είναι καλό να χρησιμοποιείται μεγάλο πλήθος επεισοδίων αξιολόγησης, ώστε τα αποτελέσματα αυτής να θεωρούνται αξιόπιστα και οι επιδόσεις των πρακτόρων σταθερές. Επομένως, στην εργασία μας, χρησιμοποιήσαμε 100 επεισόδια αξιολόγησης, τα οποία ήταν σταθερά για όλους τους αλγορίθμους.

Στις επόμενες ενότητες, θα πραγματοποιήσουμε τη σύγκριση των αλγορίθμων, πρώτα όσον αφορά το χρόνο εκπαίδευσης (*Ενότητα @sec:evaluation:time*) και στη συνέχεια όσον αφορά την τελική επίδοση τους (*Ενότητα @sec:evaluation:performance*). Τέλος, θα παρουσιάσουμε τα τελικά συμπεράσματα από την εργασία μας (*Ενότητα @sec:evaluation:conclusions*).