/* This file contains all the Bloch-PK simulators written in C. 
 * Written by Ryan Boyce 12/28/2023
 * Compile with the following command:
 * $ cc -fPIC -shared -o sims.so simulators.c
 */


#include <stdio.h>
#include <stdlib.h>
#include <math.h>

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

c_result P2L2(pk_params parms, acq_params fdv);

c_result P2L3(pk_params parms, acq_params fdv);