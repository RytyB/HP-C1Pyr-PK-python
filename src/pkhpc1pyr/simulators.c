/* This file contains all the Bloch-PK simulators written in C. 
 * Written by Ryan Boyce 12/28/2023
 * Compile with the following command:
 * $ cc -fPIC -shared -o sims.so simulators.c
 */


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_linalg.h>

#ifndef M_PI
#define M_PI (3.14159265358979323846)
#endif
double deg_to_rad(double deg){
    double rad = deg * (M_PI / 180);
    return rad;
}


typedef struct c_results{
    int *shape_factor;
    double *Mxy;
    double *Mz;
    int *sep_ind;
    double *totPyrSig;
    double *totLacSig;
}c_result;

void init_result(c_result *result, int ntp, int n_comp){
    result->shape_factor = (int*) malloc(5*sizeof(int));
    result->Mxy = (double*) malloc(n_comp*ntp*sizeof(double));
    result->Mz = (double*) malloc(n_comp*ntp*sizeof(double));
    result->sep_ind = (int*) malloc(n_comp*sizeof(int));
    result->totPyrSig = (double*) malloc(ntp*sizeof(double));
    result->totLacSig = (double*) malloc(ntp*sizeof(double));

    result->shape_factor[0] = ntp*n_comp;
    result->shape_factor[1] = ntp*n_comp;
    result->shape_factor[2] = n_comp;
    result->shape_factor[3] = ntp;
    result->shape_factor[4] = ntp;

    return;
}

void free_results(c_result result){
    free(result.shape_factor);
    free(result.Mxy);
    free(result.Mz);
    free(result.sep_ind);
    free(result.totPyrSig);
    free(result.totLacSig);
    return;
}


typedef struct pk_param_sets{
    double kpl, klp, kve, kecp, kecl;
    double vb, vef;
    double R1pyr, R1lac;
    double Pi0, Pe0, Li0, Le0;
    double *PIF;
}pk_params;


typedef struct acquisition_param_sets{
    int ntp;
    double *FA;
    double *TR;
}acq_params;


c_result P2L1(pk_params parms, acq_params fdv){ 
    int n_comp = 2; // compartments for each metabolite

    // Initialize return variables
    c_result result;
    init_result(&result, fdv.ntp, n_comp);
    int* out_shape = result.shape_factor;
    double* Mxy = result.Mxy;
    double* Mz = result.Mz;
    int* sep_ind = result.sep_ind;
    double* pyrSig = result.totPyrSig;
    double* lacSig = result.totLacSig; 

    double *Pxy = parms.PIF;
    double Pz[fdv.ntp];
    
    for (int ii=0; ii<fdv.ntp; ii++){
        // Iterate through Pz and calculate Pxy = Pz * sin( \theta )
        Pz[ii] = Pxy[ii] / sin( deg_to_rad(fdv.FA[ii]) );
    }
    double alpha_l = - (parms.klp + parms.R1lac);

    double Lzic = parms.Li0;
    for (int i=0; i<fdv.ntp; i++){
        // Calculate data for this time point
        double Lz = Lzic;
        double Lxy = Lz * sin( deg_to_rad(fdv.FA[i]) );

        // If there will be a next time point, initialize it
        if (i<fdv.ntp){
            // Signal loss due to Lac concentration
            double Lz1 = exp(alpha_l * fdv.TR[i]) * Lzic * cos( deg_to_rad(fdv.FA[i]) );

            double m = (Pz[i+1] - Pz[i])/ fdv.TR[i]; // Check these
            double b = Pz[i];

            // Additional signal being converted from Pyr
            // https://www.sfu.ca/math-coursenotes/Math%20158%20Course%20Notes/sec_first_order_homogeneous_linear.html
            // Some kind of numerical approx of an integral
            //  using integration by parts and a linear estimate of P(t)
            double Lz2 = ((m/alpha_l + b)*exp(alpha_l * fdv.TR[i]) - (m*(fdv.TR[i]+1/alpha_l))-b)/alpha_l;

            Lzic = Lz1 + parms.kpl*Lz2;
        }
        
        result.Mxy[i] = Pxy[i];
        result.Mxy[i+fdv.ntp] = Lxy;
        result.Mz[i] = Pz[i];
        result.Mz[i+fdv.ntp] = Lz;

        result.totPyrSig[i] = Pz[i];
        result.totLacSig[i] = Lz;
    }

    // Create values for sep_ind to
    //  tell where one chemical pool begins and ends
    for (int i=0; i<n_comp; i++){
        result.sep_ind[i] = i*fdv.ntp;
    }

    return result;
}

c_result P2L2(pk_params parms, acq_params fdv){

    int n_comp = 4;

    c_result result;
    init_result(&result, fdv.ntp, n_comp);
    int* out_shape = result.shape_factor;
    double* Mxy = result.Mxy;
    double* Mz = result.Mz;
    int* sep_ind = result.sep_ind;
    double* pyrSig = result.totPyrSig;
    double* lacSig = result.totLacSig; 

    // Unpack volume fractions
    double vb = parms.vb;
    double ve = 1 - parms.vb;

    double kvedve = parms.kve / ve;

    double* Pziv = parms.PIF;
    double Pxyiv[fdv.ntp];
    for (int i=0; i<fdv.ntp; i++){
        Pxyiv[i] = Pziv[i] * sin(fdv.FA[i]);
    }

    double a11 = - ( kvedve + parms.kpl + parms.R1pyr );
    double a12 = parms.klp;
    double a21 = parms.kpl;
    double a22 = - ( kvedve + parms.klp + parms.R1lac );
    
    // Use gsl library to compute matrix algebra
    gsl_matrix *A = gsl_matrix_alloc(2,2);
    gsl_vector *dD = gsl_vector_alloc(2); // Note: there's no guaruntee that the eigenvalues are real so maybe this is a problem
    gsl_matrix *P = gsl_matrix_alloc(2,2);
    gsl_eigen_nonsymm_workspace *workspace = gsl_eigen_nonsymm_alloc(2);
    gsl_matrix_set(A, 0,0, a11);
    gsl_matrix_set(A, 0,1, a12);
    gsl_matrix_set(A, 1,0, a21);
    gsl_matrix_set(A, 1,1, a22);
    gsl_eigen_nonsymm_Z(A, dD, P, workspace);

    // Setting initial conditions for acq loop
    gsl_vector* MzevIc = gsl_vector_alloc(2);
    gsl_vector* MxyevIc = gsl_vector_alloc(2);
    gsl_vector_set(MzevIc, 0, parms.Pe0);
    gsl_vecotr_set(MzevIc, 1, parms.Le0);

    // Allocating space for vectors and matrices used in acq loop
    gsl_vector* Mzev1 = gsl_vector_alloc(2);
    gsl_vector* Mzev2 = gsl_vector_alloc(2);
    for (int i=0; i<fdv.ntp; i++){

        double PxySeg = gsl_vector_get(MzevIc, 0) * sin(fdv.FA[i]);
        double LxySeg = gsl_vector_get(MzevIc, 1) * sin(fdv.FA[i]);
        gsl_vector_set(MxyevIc, 0, PxySeg);
        gsl_vector_set(MxyevIc, 1, LxySeg);

        // Check to make sure that there is a next time point to calculate
        if (i<fdv.ntp){
            
            /* CALCULATE THE FIRST PIECE OF THE SIGNAL
            MzevSeg1 = np.exp(dD*TR).flatten() * hf.mrdivide(P, np.multiply(
                        MzevSegIC, np.cos( np.radians(fdv['FlipAngle'][:, iSeg]) )
                    ).T ).flatten()
            */
            gsl_vector* term1 = gsl_vector_alloc(2);
            for (int ii=0; ii<gsl_vector_size(dD); ii++){
                gsl_vector_set(term1, ii, exp(fdv.TR[i]*gsl_vector_get(dD, ii)));
            } // This sets the first term in the dot product
            gsl_vector* term2 = gsl_vector_alloc(2);
            gsl_vector* very_temp = gsl_vector_alloc(2);
            gsl_vector_memcpy(very_temp, MzevIc);
            gsl_vector_scale(very_temp, cos(fdv.FA[i]));
            gsl_permutation* perm = gsl_permutation_alloc(2);
            int* signum;
            gsl_linalg_LU_decomp(); 
            gsl_linalg_LU_solve(); // ChatGPT thinks this is right, but I'm not convinced. needs checking

            gsl_permutation_free(perm);
            gsl_vector_free(very_temp);
            gsl_vector_free(term2);
            gsl_vector_free(term1);

            /* CALCULATE THE PORTION FROM MEMBRANE TRANSFER
            MevSeg2a = - (b.flatten() / dD) * (1-np.exp(dD*TR))
            MevSeg2b = np.multiply(m.flatten(),
                (-np.divide(TR, dD)- ((1/dD)/dD) ) + (np.exp(dD*TR) * ((1/dD)/dD)) )
            */

           // gsl_blas_ddot(P, (Mzev1 + kvedve * (Mzev2a+Mzev2b)) );

        }


    }


    gsl_vector_free(MzevIc);
    gsl_vector_free(MxyevIc);
    gsl_vector_free(Mzev1);
    gsl_vector_free(Mzev2); 
    
    gsl_matrix_free(A);
    gsl_vector_free(dD);
    gsl_eigen_nonsymm_free(workspace); 

    return result;
};

c_result P2L3(pk_params parms, acq_params fdv);