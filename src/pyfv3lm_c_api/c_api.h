
struct PyFV3LM_GridBounds_A {
    int Is, Ie, Js, Je;
    int Isd, Ied, Jsd, Jed;
    int Isc, Iec, Jsc, Jec;
    int ng;
};

extern void pyfv3lm_fv_dynamics_r8(int npx, int npy, int npz, int nq_tot, int ng, double bdt, double consv_te, 
                                   _Bool fill, _Bool reproduce_sum, double kappa, double cp_air, double zvir, 
                                   double ptop, int ks, int ncnst, int n_split, int q_split, double* u, double* v,
                                   double* w, double* delz, _Bool hydrostatic, double* pt, double* delp, double* q,
                                   double* ps, double* pe, double* pk, double* peln, double* pkz, double* phis,
                                   double* q_con, double* omga, double* ua, double* va, double* uc, double* vc,
                                   double* ak, double* bk, double* mfx, double* mfy, double* cx, double* cy,
                                   double* ze0, _Bool hybrid_z, int* gridstruct, int* flagstruct, int* neststruct, 
                                   int* idiag, struct PyFV3LM_GridBounds_A* bd, int* parent_grid, int* domain,
                                   double time_total);