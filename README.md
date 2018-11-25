# Tema3 SD (Character Recognition) - 2018

## Algoritm de rezolvare:

### Etapa de învățare:

- în funcția build din RandomForest se antrenează fiecare dintre cei 10
arbori de decizie cu câte un sample selectat aleator din cele de input

- în funcția train din DecisionTree se verifică inițial dacă splitul conține
o singură clasă, caz în care se creează o frunză, al cărei rezultat va fi
respectiva clasă;

- apoi, se determină split-ul ce maximizează inofrmation gain-ul pentru un
set de dimensiuni aleatoare din sample-ul primit ca paramteru;

- pentru a se afla acest split optim, în funcția find_best_split se vor
parcurge elementele unicizate ale fiecărei dimensiuni din vectorul dimensions,
calculându-se la fiecare pas entropiile copiilor generați dacă s-ar considera
split_index ca fiind dimensiunea curentă și split_value ca fiind elementul
curent din vectorul unicizat;

- se returnează perechea de split_index și split_value care obține un
information gain maxim;

- în cazul în care find_best_split întoarce un split invalid, se creeaza o
frunză în care result va fi clasa cel mai des întâlnită în sample-ul curent;

- altfel, se creează split-ul ce corespunde perechii menționate, precum și
nodurile stâng și drept și algoritmul continuă recursiv;

### Etapa de predicție:

- în funcția predict din RandomForest se determină, cu ajutorul unui vector
de frecvență, clasa ce este prezisă de cei mai mulți arbori de decizie;

- în funcția omonimă din DecisionTree se caută și, ulterior, se returnează
clasa după modelul unui arbore binar de căutare: dacă s-a gasit o frunză,
valoarea lui result din aceasta este clasa cautată, iar dacă nu, se va căuta o
frunză în stânga sau în dreapta, în funcție de split_index și split_value.
