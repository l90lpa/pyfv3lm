module pyfv3lm_mod
    use iso_c_binding
    implicit none

    type, bind(c) :: PyFV3LM_GridBounds_A
        integer(c_int) :: Is, Ie, Js, Je,     &
                          Isd, Ied, Jsd, Jed, &
                          Isc, Iec, Jsc, Jec, &
                          ng
    end type


    interface
        subroutine pyfv3lm_fv_dynamics_r8(npx, npy, npz, nq_tot, ng, bdt, consv_te,          &
                                          fill, reproduce_sum, kappa, cp_air, zvir,          &
                                          ptop, ks, ncnst, n_split, q_split, u, v,           &
                                          w, delz, hydrostatic, pt, delp, q,                 &
                                          ps, pe, pk, peln, pkz, phis,                       &
                                          q_con, omga, ua, va, uc, vc,                       &
                                          ak, bk, mfx, mfy, cx, cy,                          &
                                          ze0, hybrid_z, gridstruct, flagstruct, neststruct, & 
                                          idiag, bd, parent_grid, domain,                    &
                                          time_total) bind(c, name="pyfv3lm_fv_dynamics_r8")
            use iso_c_binding
            import :: PyFV3LM_GridBounds_A

            real(c_double), value, intent(in) :: bdt  ! Large time-step
            real(c_double), value, intent(in) :: consv_te
            real(c_double), value, intent(in) :: kappa, cp_air
            real(c_double), value, intent(in) :: zvir, ptop
            real(c_double), value, intent(in) :: time_total
        
            integer(c_int), value, intent(in) :: npx
            integer(c_int), value, intent(in) :: npy
            integer(c_int), value, intent(in) :: npz
            integer(c_int), value, intent(in) :: nq_tot             ! transported tracers
            integer(c_int), value, intent(in) :: ng
            integer(c_int), value, intent(in) :: ks
            integer(c_int), value, intent(in) :: ncnst
            integer(c_int), value, intent(in) :: n_split        ! small-step horizontal dynamics
            integer(c_int), value, intent(in) :: q_split        ! tracer
            logical(c_bool), value, intent(in) :: fill
            logical(c_bool), value, intent(in) :: reproduce_sum
            logical(c_bool), value, intent(in) :: hydrostatic
            logical(c_bool), value, intent(in) :: hybrid_z       ! Using hybrid_z for remapping
        
            type(PyFV3LM_GridBounds_A), intent(in) :: bd
            real(c_double), intent(inout), dimension(bd%isd:bd%ied  ,bd%jsd:bd%jed+1,npz) :: u ! D grid zonal wind (m/s)
            real(c_double), intent(inout), dimension(bd%isd:bd%ied+1,bd%jsd:bd%jed  ,npz) :: v ! D grid meridional wind (m/s)
            real(c_double), intent(inout) :: w(   bd%isd:  ,bd%jsd:  ,1:)  !  W (m/s)
            real(c_double), intent(inout) :: pt(  bd%isd:bd%ied  ,bd%jsd:bd%jed  ,npz)  ! temperature (K)
            real(c_double), intent(inout) :: delp(bd%isd:bd%ied  ,bd%jsd:bd%jed  ,npz)  ! pressure thickness (pascal)
            real(c_double), intent(inout) :: q(   bd%isd:bd%ied  ,bd%jsd:bd%jed  ,npz, ncnst) ! specific humidity and constituents
            real(c_double), intent(inout) :: delz(bd%isd:,bd%jsd:,1:)   ! delta-height (m); non-hydrostatic only
            real(c_double), intent(inout) ::  ze0(bd%is:, bd%js: ,1:) ! height at edges (m); non-hydrostatic ! ze0 no longer used
        
        !-----------------------------------------------------------------------
        ! Auxilliary pressure arrays:    
        ! The 5 vars below can be re-computed from delp and ptop.
        !-----------------------------------------------------------------------
        ! dyn_aux:
            real(c_double), intent(inout) :: ps  (bd%isd:bd%ied  ,bd%jsd:bd%jed)           ! Surface pressure (pascal)
            real(c_double), intent(inout) :: pe  (bd%is-1:bd%ie+1, npz+1,bd%js-1:bd%je+1)  ! edge pressure (pascal)
            real(c_double), intent(inout) :: pk  (bd%is:bd%ie,bd%js:bd%je, npz+1)          ! pe**kappa
            real(c_double), intent(inout) :: peln(bd%is:bd%ie,npz+1,bd%js:bd%je)           ! ln(pe)
            real(c_double), intent(inout) :: pkz (bd%is:bd%ie,bd%js:bd%je,npz)             ! finite-volume mean pk
            real(c_double), intent(inout):: q_con(bd%isd:, bd%jsd:, 1:)
            
        !-----------------------------------------------------------------------
        ! Others:
        !-----------------------------------------------------------------------
            real(c_double), intent(inout) :: phis(bd%isd:bd%ied,bd%jsd:bd%jed)       ! Surface geopotential (g*Z_surf)
            real(c_double), intent(inout) :: omga(bd%isd:bd%ied,bd%jsd:bd%jed,npz)   ! Vertical pressure velocity (pa/s)
            real(c_double), intent(inout) :: uc(bd%isd:bd%ied+1,bd%jsd:bd%jed  ,npz) ! (uc,vc) mostly used as the C grid winds
            real(c_double), intent(inout) :: vc(bd%isd:bd%ied  ,bd%jsd:bd%jed+1,npz)
            real(c_double), intent(inout), dimension(bd%isd:bd%ied ,bd%jsd:bd%jed ,npz):: ua, va
            real(c_double), intent(in),    dimension(npz+1):: ak, bk
        
        ! Accumulated Mass flux arrays: the "Flux Capacitor"
            real(c_double), intent(inout) ::  mfx(bd%is:bd%ie+1, bd%js:bd%je,   npz)
            real(c_double), intent(inout) ::  mfy(bd%is:bd%ie  , bd%js:bd%je+1, npz)
        ! Accumulated Courant number arrays
            real(c_double), intent(inout) ::  cx(bd%is:bd%ie+1, bd%jsd:bd%jed, npz)
            real(c_double), intent(inout) ::  cy(bd%isd:bd%ied ,bd%js:bd%je+1, npz)
        
            integer(c_int), dimension(2), intent(inout) :: gridstruct
            integer(c_int), dimension(2), intent(inout) :: flagstruct
            integer(c_int), dimension(2), intent(inout) :: neststruct
            integer(c_int), dimension(2), intent(inout) :: domain
            integer(c_int), dimension(2), intent(inout) :: parent_grid
            integer(c_int), dimension(2), intent(in)    :: idiag
        end subroutine
    end interface
    
contains

end module pyfv3lm_mod