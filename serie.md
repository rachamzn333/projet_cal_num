# README : solution de la serie 

# Exercice 3

---

## 1. Comment déclarer et allouer une matrice pour utiliser BLAS et LAPACK ?

Pour utiliser BLAS et LAPACK, une matrice doit être déclarée comme un tableau unidimensionnel en mémoire contiguë, avec un stockage en colonne majeure ou en ligne majeure.

### Exemple de déclaration :
```c
#include <stdlib.h>

double* matrix = (double*)malloc(rows * cols * sizeof(double));
```
- `rows` : nombre de lignes.
- `cols` : nombre de colonnes.

### Accès aux éléments :
- **Colonne majeure (LAPACK)** : `matrix[i + j * rows]`.
- **Ligne majeure (BLAS optionnel)** : `matrix[i * cols + j]`.

### Libération de la mémoire :
```c
free(matrix);
```

---

## 2. Qu'est-ce que la constante `LAPACK_COL_MAJOR` ?

La constante `LAPACK_COL_MAJOR` indique que les matrices sont stockées en **colonne majeure** dans la mémoire. Cela signifie que les éléments d'une colonne sont contigus.

### Exemple :
Pour une matrice 3x3 :
```
1 4 7
2 5 8
3 6 9
```
Elle est stockée en mémoire comme : `[1, 2, 3, 4, 5, 6, 7, 8, 9]`.

---

## 3. Qu'est-ce que la dimension principale (`leading dimension` ou `ld`) ?

La dimension principale (ou `ld`) représente le nombre d'éléments stockés dans une colonne (pour le stockage en colonne majeure) ou une ligne (pour le stockage en ligne majeure).

- En colonne majeure, `ld` correspond au nombre de lignes physiques dans la mémoire.

### Exemple :
Pour une matrice \(4 \times 3\) en colonne majeure, `ld = 4`.

---

## 4. Fonction `dgbmv` : Produit matrice-vecteur

La fonction `dgbmv` effectue le produit matrice-vecteur pour une matrice en stockage bande.

### Prototype :
```c
void cblas_dgbmv(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE Trans,
                 const int M, const int N, const int KL, const int KU,
                 const double alpha, const double *A, const int lda,
                 const double *X, const int incX, const double beta,
                 double *Y, const int incY);
```

### Fonctionnement :
Elle calcule :
\[
Y = \alpha \cdot A \cdot X + \beta \cdot Y
\]
- `A` : matrice bande.
- `KL` et `KU` : nombres de sous-diagonales et sur-diagonales de `A`.
- `X` et `Y` : vecteurs.

---

## 5. Fonction `dgbtrf` : Factorisation LU

Cette fonction effectue une factorisation LU partielle d'une matrice bande.

### Prototype :
```c
void dgbtrf_(const int* m, const int* n, const int* kl, const int* ku,
             double* ab, const int* ldab, int* ipiv, int* info);
```

### Fonctionnement :
Elle décompose la matrice \(A\) en :
\[
A = P \cdot L \cdot U
\]
- `L` : matrice triangulaire inférieure.
- `U` : matrice triangulaire supérieure.
- `P` : matrice de permutation.
- `ipiv` : indices de pivot.

---

## 6. Fonction `dgbtrs` : Résolution avec factorisation LU

Cette fonction résout un système linéaire à l'aide de la factorisation LU calculée par `dgbtrf`.

### Prototype :
```c
void dgbtrs_(const char* trans, const int* n, const int* kl, const int* ku,
             const int* nrhs, const double* ab, const int* ldab,
             const int* ipiv, double* b, const int* ldb, int* info);
```

### Fonctionnement :
Elle résout :
\[
AX = B
\]
- `A` : factorisé en LU par `dgbtrf`.
- `ipiv` : tableau de pivots.

---

## 7. Fonction `dgbsv` : Résolution directe

Cette fonction combine les étapes de `dgbtrf` et `dgbtrs` pour résoudre directement un système linéaire.

### Prototype :
```c
void dgbsv_(const int* n, const int* kl, const int* ku, const int* nrhs,
            double* ab, const int* ldab, int* ipiv, double* b,
            const int* ldb, int* info);
```

### Fonctionnement :
Elle calcule directement la solution \(X\) dans :
\[
AX = B
\]
- `A` : matrice bande.
- Combine factorisation LU et résolution.

---

## 8. Calcul de la norme du résidu relatif

Le résidu relatif mesure la précision d'une solution calculée.

### Formule :
\[
\text{Residue} = \frac{\|B - AX\|_2}{\|B\|_2}
\]

### Exemple de code :
```c
#include <cblas.h>
double compute_residual(const double* A, const double* X, const double* B,
                        int n, int ldA) {
    double* AX = (double*)malloc(n * sizeof(double));
    cblas_dgbmv(CblasColMajor, CblasNoTrans, n, n, kl, ku, 1.0, A, ldA, X, 1, 0.0, AX, 1);

    for (int i = 0; i < n; i++) AX[i] = B[i] - AX[i];

    double res_norm = cblas_dnrm2(n, AX, 1);
    double b_norm = cblas_dnrm2(n, B, 1);

    free(AX);
    return res_norm / b_norm;
}
```

---
# Exercice 4 : Stockage GB et Appel à DGBMV

---



### 1. Écriture du stockage GB pour la matrice de Poisson 1D

Le stockage GB (General Band) est utilisé pour des matrices tridiagonales dans l'équation de Poisson 1D. Pour créer cette matrice dans le code, nous utilisons la fonction `set_GB_operator_colMajor_poisson1D` déjà implémentée dans `lib_poisson1D.c`. Cette fonction initialise la matrice bande avec :

- **Diagonale principale** : Valeurs à -2.0.
- **Sous-diagonale et sur-diagonale** : Valeurs à 1.0.

#### Exemple de fonction :
```c
void set_GB_operator_colMajor_poisson1D(double* AB, int *lab, int *la, int *kv){
    int i, j;
    int kl = 1; // Sous-diagonale
    int ku = 1; // Sur-diagonale
    int ldab = kl + ku + 1; // Taille réelle des lignes dans le tableau GB

    for (j = 0; j < *la; j++) {
        for (i = 0; i < ldab; i++) {
            AB[i + j * ldab] = 0.0; // Initialisation
        }
    }

    for (j = 0; j < *la; j++) {
        AB[ku + j * ldab] = -2.0;         // Diagonale principale
        if (j > 0) AB[ku - 1 + j * ldab] = 1.0;  // Sous-diagonale
        if (j < *la - 1) AB[ku + 1 + j * ldab] = 1.0;  // Sur-diagonale
    }
}
```

---

### 2. Utilisation de la fonction BLAS `dgbmv`

La fonction `dgbmv` permet de réaliser le produit matrice-vecteur pour une matrice en stockage bande.

#### Exemple de code utilisant `dgbmv` :
```c
#include <cblas.h>

void test_dgbmv_poisson1D(double* AB, double* X, double* Y, int la) {
    int kl = 1, ku = 1;
    int ldab = kl + ku + 1;

    cblas_dgbmv(CblasColMajor, CblasNoTrans, la, la, kl, ku, 1.0, AB, ldab, X, 1, 0.0, Y, 1);
}
```
- `AB` : Matrice en format GB.
- `X` : Vecteur d'entrée.
- `Y` : Vecteur résultat.
- `kl` et `ku` : Nombres de sous-diagonales et sur-diagonales.

---

### 3. Méthode de validation

Pour valider le résultat :

1. **Initialisez** la matrice `AB` avec `set_GB_operator_colMajor_poisson1D`.
2. **Définissez** un vecteur d'entrée simple.
3. **Calculez manuellement** le produit pour une matrice tridiagonale simple.
4. **Comparez** le résultat obtenu par `dgbmv` avec le calcul manuel.
5. **Affichez l'erreur** (si présente) pour vérifier la validité du produit matrice-vecteur.

---



# exercice 5


---

## 1. Résolution par une Méthode Directe avec LAPACK

Pour résoudre un système linéaire \(AX = B\) où \(A\) est une matrice bande, on utilise :

- **`dgbtrf`** : Factorisation LU de la matrice bande \(A\).
- **`dgbtrs`** : Résolution du système en utilisant la factorisation LU.

### Exemple d’Implémentation en C :

#### Code Complet :
```c
#include <stdio.h>
#include <lapacke.h>

void solve_poisson1D(double* AB, double* B, int la, int kl, int ku) {
    int ldab = 2 * kl + ku + 1; // Taille de la bande stockée
    int ipiv[la]; // Tableau pour les indices de pivotage
    int info;

    // Factorisation LU
    LAPACKE_dgbtrf(LAPACK_COL_MAJOR, la, la, kl, ku, AB, ldab, ipiv);

    // Résolution du système
    LAPACKE_dgbtrs(LAPACK_COL_MAJOR, 'N', la, kl, ku, 1, AB, ldab, ipiv, B, la);
}

int main() {
    int la = 5;
    int kl = 1, ku = 1;
    double AB[3 * 5] = {0};
    double B[5] = {1, 2, 3, 4, 5};

    set_GB_operator_colMajor_poisson1D(AB, &(int){3}, &la, &(int){1});
    solve_poisson1D(AB, B, la, kl, ku);

    printf("Solution X:\n");
    for (int i = 0; i < la; i++) {
        printf("%f\n", B[i]);
    }
    return 0;
}
```

---

## 2. Complexité des Algorithmes

### A. Complexité de `dgbtrf` (Factorisation LU)

La routine `dgbtrf` réalise la factorisation LU d'une matrice bande. La complexité est proportionnelle au nombre de flops nécessaires pour réduire chaque colonne, en tenant compte des bandes.

#### Formule :
\[
\text{Complexité} = N \cdot (2 \cdot KL \cdot KU)
\]
- \(N\) : Taille de la matrice.
- \(KL\) : Nombre de sous-diagonales.
- \(KU\) : Nombre de sur-diagonales.

#### Exemple :
Pour une matrice tridiagonale (\(KL = KU = 1\)) de taille \(N = 1000\) :
\[
\text{Complexité} = 1000 \cdot (2 \cdot 1 \cdot 1) = 2000 \text{ flops.}
\]

---

### B. Complexité de `dgbtrs` (Résolution du Système)

Après la factorisation LU, la résolution du système avec `dgbtrs` est proportionnelle au nombre d'éléments dans la bande.

#### Formule :
\[
\text{Complexité} = N \cdot (KL + KU)
\]

#### Exemple :
Pour une matrice tridiagonale (\(KL = KU = 1\), \(N = 1000\)) :
\[
\text{Complexité} = 1000 \cdot (1 + 1) = 2000 \text{ flops.}
\]

---

### C. Complexité Totale (\(dgbtrf + dgbtrs\))

La complexité totale est la somme des étapes de factorisation et de résolution :
\[
\text{Complexité Totale} = N \cdot (2 \cdot KL \cdot KU) + N \cdot (KL + KU)
\]

#### Exemple :
Pour une matrice tridiagonale (\(KL = KU = 1\), \(N = 1000\)) :
- **Factorisation LU (`dgbtrf`)** : 2000 flops.
- **Résolution (`dgbtrs`)** : 2000 flops.
\[
\text{Complexité Totale} = 2000 + 2000 = 4000 \text{ flops.}
\]

#### Validation pour des Matrices Plus Larges :
Pour une matrice bande avec \(KL = KU = 5\) et \(N = 1000\) :
- **Factorisation LU (`dgbtrf`)** :
\[
1000 \cdot (2 \cdot 5 \cdot 5) = 50,000 \text{ flops.}
\]
- **Résolution (`dgbtrs`)** :
\[
1000 \cdot (5 + 5) = 10,000 \text{ flops.}
\]
\[
\text{Complexité Totale} = 50,000 + 10,000 = 60,000 \text{ flops.}
\]

---
#exercice 6
---

## 1. Factorisation LU pour Matrices Tridiagonales

### Implémentation en C
La factorisation LU décompose une matrice \(A\) en deux matrices :
- \(L\) : matrice triangulaire inférieure avec des 1 sur la diagonale.
- \(U\) : matrice triangulaire supérieure.

#### Code C :
```c
#include <stdio.h>
#include <stdlib.h>

void LU_factorization_tridiagonal(double* AB, int N) {
    int i;
    for (i = 1; i < N; i++) {
        // Calcul du pivot (L[i, i-1])
        AB[N + i] = AB[N + i] / AB[i - 1];

        // Mise à jour de la diagonale principale (U[i, i])
        AB[i] = AB[i] - AB[N + i] * AB[i - 1 + N];
    }
}

void print_matrix(double* AB, int N) {
    printf("Matrice LU (format GB):\n");
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < N; j++) {
            printf("%.2f ", AB[i + j * 3]);
        }
        printf("\n");
    }
}

int main() {
    int N = 5;
    double AB[3 * 5] = {
        0, -1, -1, -1, -1,  // Sous-diagonale
        2,  2,  2,  2,  2,  // Diagonale principale
       -1, -1, -1, -1,  0   // Sur-diagonale
    };

    LU_factorization_tridiagonal(AB, N);
    print_matrix(AB, N);

    return 0;
}
```

---

## 2. Validation de la Factorisation LU

### Méthode de Validation
Pour valider la factorisation LU :
1. **Extraire** \(L\) et \(U\) du tableau compact \(AB\).
2. **Reconstituer** \(A\) en calculant \(L \cdot U\).
3. **Calculer l’erreur** en comparant \(A\) et \(L \cdot U\).

#### Code C :
```c
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

void validate_LU(double* AB, int N) {
    double* L = (double*)calloc(N * N, sizeof(double));
    double* U = (double*)calloc(N * N, sizeof(double));
    double* A_reconstructed = (double*)calloc(N * N, sizeof(double));

    // Extraction de L et U
    for (int j = 0; j < N; j++) {
        L[j + j * N] = 1.0; // Diagonale de L
        if (j > 0) {
            L[j + (j - 1) * N] = AB[N + j];
        }
        U[j + j * N] = AB[j];
        if (j < N - 1) {
            U[j + (j + 1) * N] = AB[j + 1];
        }
    }

    // Reconstituer A = L * U
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                A_reconstructed[i + j * N] += L[i + k * N] * U[k + j * N];
            }
        }
    }

    // Calculer l'erreur ||A - L*U||
    double error = 0.0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double diff = A_reconstructed[i + j * N] - AB[i + j * 3];
            error += diff * diff;
        }
    }
    error = sqrt(error);

    printf("Erreur de reconstruction : %e\n", error);

    free(L);
    free(U);
    free(A_reconstructed);
}
```

---

## 3. Résolution Itérative avec la Méthode de Jacobi

### Principe de la Méthode de Jacobi
La méthode de Jacobi résout un système linéaire \(AX = B\) en construisant une suite d'approximations X^(k).
### Implémentation en C avec BLAS
X_i^(k+1) = (1 / A_ii) * (B_i - Σ (A_ij * X_j^(k)) pour j ≠ i)
#### Code C :
```c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cblas.h>

void jacobi_method(double* A, double* B, double* X, int n, double tol, int maxit) {
    double* X_new = (double*)malloc(n * sizeof(double));
    double* res = (double*)malloc(n * sizeof(double));
    int iter = 0;

    // Initialisation de X_new à 0
    for (int i = 0; i < n; i++) {
        X_new[i] = 0.0;
    }

    while (iter < maxit) {
        // Une itération de Jacobi
        for (int i = 0; i < n; i++) {
            double sum = 0.0;
            for (int j = 0; j < n; j++) {
                if (i != j) {
                    sum += A[i * n + j] * X[j];
                }
            }
            X_new[i] = (B[i] - sum) / A[i * n + i];
        }

        // Calcul du résidu : res = B - A * X_new
        cblas_dcopy(n, B, 1, res, 1);
        cblas_dgemv(CblasRowMajor, CblasNoTrans, n, n, -1.0, A, n, X_new, 1, 1.0, res, 1);

        // Norme du résidu
        double res_norm = cblas_dnrm2(n, res, 1);
        if (res_norm < tol) {
            break;
        }

        // Mise à jour de X
        for (int i = 0; i < n; i++) {
            X[i] = X_new[i];
        }

        iter++;
    }

    if (iter == maxit) {
        printf("Méthode de Jacobi n'a pas convergé après %d itérations.\n", maxit);
    } else {
        printf("Convergence en %d itérations.\n", iter);
    }

    free(X_new);
    free(res);
}

int main() {
    int n = 5; // Taille de la matrice
    double A[25] = { // Matrice tridiagonale
        2, -1, 0, 0, 0,
       -1,  2, -1, 0, 0,
        0, -1,  2, -1, 0,
        0,  0, -1,  2, -1,
        0,  0,  0, -1,  2
    };
    double B[5] = {1, 1, 1, 1, 1}; // Second membre
    double X[5] = {0, 0, 0, 0, 0}; // Solution initiale

    double tol = 1e-6; // Tolérance
    int maxit = 100;   // Nombre d'itérations maximum

    jacobi_method(A, B, X, n, tol, maxit);

    printf("Solution :\n");
    for (int i = 0; i < n; i++) {
        printf("X[%d] = %f\n", i, X[i]);
    }

    return 0;
}
```

---
