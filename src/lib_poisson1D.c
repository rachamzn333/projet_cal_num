/**********************************************/
/* lib_poisson1D.c                            */
/* Numerical library developed to solve 1D    */ 
/* Poisson problem (Heat equation)            */
/**********************************************/
#include "lib_poisson1D.h"
#include <lapacke.h>
#include <math.h>
#include <stdio.h>

void set_GB_operator_colMajor_poisson1D(double *AB, int *lab, int *la, int *kv) {
    int i, j;

    for (j = 0; j < *la; j++) {
        for (i = 0; i < *lab; i++) {
            AB[i + j * (*lab)] = 0.0; // Initialisation à zéro
        }
        AB[*kv + j * (*lab)] = 2.0; // Diagonale principale
        if (j > 0) {
            AB[*kv - 1 + j * (*lab)] = -1.0; // Sous-diagonale
        }
        if (j < *la - 1) {
            AB[*kv + 1 + j * (*lab)] = -1.0; // Sur-diagonale
        }
    }
}
void set_GB_operator_colMajor_poisson1D_Id(double* AB, int *lab, int *la, int *kv) {
    int i, j;

    // Initialiser la matrice bande AB à zéro
    for (j = 0; j < *la; j++) {
        for (i = 0; i < *lab; i++) {
            AB[i + j * (*lab)] = 0.0;
        }
    }

    // Remplir la diagonale principale avec 1.0
    for (j = 0; j < *la; j++) {
        AB[*kv + j * (*lab)] = 1.0;
    }

    // Remplir les sous-diagonales et sur-diagonales avec 0.0 (déjà initialisé à zéro)
}





void set_dense_RHS_DBC_1D(double *RHS, int *la, double *T0, double *T1) {
    for (int i = 0; i < *la; i++) {
        RHS[i] = 0.0; // Initialise à zéro
    }
    RHS[0] -= *T0;       // Bord gauche
    RHS[*la - 1] -= *T1; // Bord droit
}
 

void set_analytical_solution_DBC_1D(double *EX_SOL, double *X, int *la, double *T0, double *T1) {
    for (int i = 0; i < *la; i++) {
        EX_SOL[i] = *T0 + X[i] * (*T1 - *T0); // Solution linéaire pour test
    }
}
 

void set_grid_points_1D(double *X, int *la) {
    double dx = 1.0 / (*la + 1);
    for (int i = 0; i < *la; i++) {
        X[i] = (i + 1) * dx;
    }
}



double relative_forward_error(double* x, double* y, int* la) {
    double norm_diff = 0.0;
    double norm_x = 0.0;

    for (int i = 0; i < *la; i++) {
        norm_diff += (x[i] - y[i]) * (x[i] - y[i]);
        norm_x += x[i] * x[i];
    }

    norm_diff = sqrt(norm_diff);
    norm_x = sqrt(norm_x);

    if (norm_x == 0.0) {
        return -1.0; // Retourner -1 pour éviter une division par zéro
    }

    return norm_diff / norm_x; // Erreur relative
}


int indexABCol(int i, int j, int *lab) {
    return i + j * (*lab); // Index dans une matrice bande stockée en colonne-major
}


    int dgbtrftridiag(int *la, int*n, int *kl, int *ku, double *AB, int *lab, int *ipiv, int *info){
      int i;
    double l;

    for (i = 0; i < *n - 1; i++) {
        if (AB[*ku + i * (*lab)] == 0.0) {
            *info = i + 1; // Pivot nul, la factorisation échoue
            return *info;
        }

        l = AB[*ku - 1 + (i + 1) * (*lab)] / AB[*ku + i * (*lab)]; // Calcul du facteur
        AB[*ku - 1 + (i + 1) * (*lab)] = l; // Stocke L sous la diagonale principale
        AB[*ku + (i + 1) * (*lab)] -= l * AB[*ku + 1 + i * (*lab)]; // Mise à jour U
    }

    *info = 0; // Succès
  return *info;
}





