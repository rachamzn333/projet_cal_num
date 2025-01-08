/**********************************************/
/* lib_poisson1D.c                            */
/* Numerical library developed to solve 1D    */ 
/* Poisson problem (Heat equation)            */
/**********************************************/
#include "lib_poisson1D.h"

void eig_poisson1D(double* eigval, int *la) {
    for (int i = 0; i < *la; i++) {
        eigval[i] = 2.0 - 2.0 * cos((i + 1) * M_PI / (*la + 1));
    }
}


double eigmax_poisson1D(int *la) {
    return 2.0 - 2.0 * cos(M_PI * (*la) / (*la + 1));
}

double eigmin_poisson1D(int *la) {
    return 2.0 - 2.0 * cos(M_PI / (*la + 1));
}

double richardson_alpha_opt(int *la){
  return 0;
}

void richardson_alpha(double *AB, double *RHS, double *X, double *alpha_rich, int *lab, int *la, int *ku, int*kl, double *tol, int *maxit, double *resvec, int *nbite) {
    double *residual = malloc(*la * sizeof(double));
    double *Ax = malloc(*la * sizeof(double));
    *nbite = 0;

    for (int k = 0; k < *maxit; k++) {
        // Calcul de Ax
        matvec_gb(*la, AB, *lab, X, Ax);

        // Calcul du résidu
        for (int i = 0; i < *la; i++) {
            residual[i] = RHS[i] - Ax[i];
        }

        // Norme du résidu
        double res_norm = 0.0;
        for (int i = 0; i < *la; i++) {
            res_norm += residual[i] * residual[i];
        }
        res_norm = sqrt(res_norm);
        resvec[k] = res_norm;

        // Convergence
        if (res_norm < *tol) {
            *nbite = k + 1;
            break;
        }

        // Mise à jour de X
        for (int i = 0; i < *la; i++) {
            X[i] += (*alpha_rich) * residual[i];
        }
    }

    free(residual);
    free(Ax);
}


void extract_MB_jacobi_tridiag(double *AB, double *MB, int *lab, int *la, int *ku, int *kl, int *kv) {
    for (int i = 0; i < *la; i++) {
        MB[*kv + i * (*lab)] = AB[*ku + i * (*lab)];
    }
}


void extract_MB_gauss_seidel_tridiag(double *AB, double *MB, int *lab, int *la, int *ku, int *kl, int *kv) {
    for (int i = 0; i < *la; i++) {
        MB[*kv + i * (*lab)] = AB[*ku + i * (*lab)];
        if (i > 0) {
            MB[*kv - 1 + i * (*lab)] = AB[*ku - 1 + i * (*lab)];
        }
    }
}


void richardson_MB(double *AB, double *RHS, double *X, double *MB, int *lab, int *la, int *ku, int *kl, double *tol, int *maxit, double *resvec, int *nbite) {
    double *residual = malloc(*la * sizeof(double));
    double *Ax = malloc(*la * sizeof(double));
    double *M_inv_res = malloc(*la * sizeof(double));
    *nbite = 0;

    for (int k = 0; k < *maxit; k++) {
        matvec_gb(*la, AB, *lab, X, Ax); // Calcul de Ax

        // Calcul du résidu
        for (int i = 0; i < *la; i++) {
            residual[i] = RHS[i] - Ax[i];
        }

        // Résolution M * M_inv_res = residual
        for (int i = 0; i < *la; i++) {
            M_inv_res[i] = residual[i] / MB[*lab / 2 + i * (*lab)];
        }

        // Mise à jour de X
        for (int i = 0; i < *la; i++) {
            X[i] += M_inv_res[i];
        }

        // Norme du résidu
        double res_norm = 0.0;
        for (int i = 0; i < *la; i++) {
            res_norm += residual[i] * residual[i];
        }
        res_norm = sqrt(res_norm);
        resvec[k] = res_norm;

        if (res_norm < *tol) {
            *nbite = k + 1;
            break;
        }
    }

    free(residual);
    free(Ax);
    free(M_inv_res);
}

