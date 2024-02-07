module pyfv3lm_fv3jedilm_mock_mod
    use iso_c_binding
    use pyfv3lm_mod, only: pyfv3lm_fv_dynamics_r8, PyFV3LM_GridBounds_A
    use fv_arrays_nlm_mod, only: fv_grid_bounds_type, fv_grid_type, fv_flags_type, fv_nest_type, fv_atmos_type, fv_diag_type
    use fv_dynamics_nlm_mod, only: fv_dynamics
    use fv_dynamics_tlm_mod, only: fv_dynamics_tlm, fv_dynamics_nlm => fv_dynamics
    use fv_pressure_mod, only: compute_fv3_pressures
    use fv3jedi_lm_mod, only: fv3jedi_lm_type
    use fv3jedi_lm_dynamics_mod, only: fv3jedi_lm_dynamics_type
    use fv3jedi_lm_utils_mod, only: fv3jedi_lm_conf, fv3jedi_lm_traj
    use fv3jedi_lm_kinds_mod, only: kind_real
    use fv3jedi_lm_const_mod, only: cp, kappa, zvir
    use fms_io_mod, only: set_domain, nullify_domain
    use mpp_domains_mod, only: domain2d, mpp_get_boundary, mpp_update_domains
    use mpp_parameter_mod, only: DGRID_NE
    
    implicit none
    private

    ! Precision of dyncore
#ifdef SINGLE_FV
    integer, parameter :: fvprec = 4
#else
    integer, parameter :: fvprec = 8
#endif

    public :: step_nl
    
contains

    subroutine step_nl(self)

        implicit none
    
        type(fv3jedi_lm_type), intent(inout) :: self
    
        if (self%conf%do_dyn == 1) call dynamics_step_nl(self%fv3jedi_lm_dynamics,self%conf,self%traj)
        if (self%conf%do_phy == 1) call self%fv3jedi_lm_physics%step_nl(self%conf,self%traj)
    
    endsubroutine step_nl

    subroutine dynamics_step_nl(self, conf, traj)
        use iso_c_binding
        implicit none

        type(fv3jedi_lm_dynamics_type), intent(inout), target :: self
        type(fv3jedi_lm_traj), intent(inout) :: traj
        type(fv3jedi_lm_conf), intent(in) :: conf
       
        type(fv_atmos_type), pointer :: FV_Atm(:)
        integer :: i,j,k
        type(PyFV3LM_GridBounds_A) :: c_bd
        integer(c_int), dimension(2) :: grid_handle, flags_handle, nest_handle, &
                                        idiag_handle, parent_grid_handle, domain_handle
        logical(c_bool) :: c_fill, c_reproduce_sum, c_hydrostatic, c_hybrid_z
        real :: dummy_time = 3.14159
       
        !Convenience pointer to the main FV_Atm structure
        !------------------------------------------------
        FV_Atm => self%FV_Atm
       
        !Copy from traj to the fv3 structure
        !-----------------------------------
        call traj_to_fv3(self,conf,traj)
       
        ! MPP set domain
        ! --------------
        call set_domain(FV_Atm(1)%domain)
       
        !Propagate FV3 one time step
        !---------------------------
        if (self%linmodtest == 0) then
            ! call fv_dynamics( FV_Atm(1)%npx, FV_Atm(1)%npy, FV_Atm(1)%npz, FV_Atm(1)%ncnst, FV_Atm(1)%ng,  &
            !                   real(conf%dt, fvprec), FV_Atm(1)%flagstruct%consv_te, FV_Atm(1)%flagstruct%fill,           &
            !                   FV_Atm(1)%flagstruct%reproduce_sum, real(kappa, fvprec),                                   &
            !                   real(cp, fvprec), real(zvir, fvprec), FV_Atm(1)%ptop, FV_Atm(1)%ks, FV_Atm(1)%flagstruct%ncnst,          &
            !                   FV_Atm(1)%flagstruct%n_split, FV_Atm(1)%flagstruct%q_split,                  &
            !                   FV_Atm(1)%u, FV_Atm(1)%v, FV_Atm(1)%w, FV_Atm(1)%delz,                       &
            !                   FV_Atm(1)%flagstruct%hydrostatic, FV_Atm(1)%pt, FV_Atm(1)%delp, FV_Atm(1)%q, &
            !                   FV_Atm(1)%ps, FV_Atm(1)%pe, FV_Atm(1)%pk, FV_Atm(1)%peln, FV_Atm(1)%pkz,     &
            !                   FV_Atm(1)%phis, FV_Atm(1)%q_con, FV_Atm(1)%omga,                             &
            !                   FV_Atm(1)%ua, FV_Atm(1)%va, FV_Atm(1)%uc, FV_Atm(1)%vc,                      &
            !                   FV_Atm(1)%ak, FV_Atm(1)%bk,                                                  &
            !                   FV_Atm(1)%mfx, FV_Atm(1)%mfy, FV_Atm(1)%cx, FV_Atm(1)%cy, FV_Atm(1)%ze0,     &
            !                   FV_Atm(1)%flagstruct%hybrid_z, FV_Atm(1)%gridstruct, FV_Atm(1)%flagstruct,   &
            !                   FV_Atm(1)%neststruct, FV_Atm(1)%idiag, FV_Atm(1)%bd, FV_Atm(1)%parent_grid,  &
            !                   FV_Atm(1)%domain )

            c_fill = FV_Atm(1)%flagstruct%fill
            c_reproduce_sum = FV_Atm(1)%flagstruct%reproduce_sum
            c_hydrostatic = FV_Atm(1)%flagstruct%hydrostatic
            c_hybrid_z = FV_Atm(1)%flagstruct%hybrid_z
            
            c_bd = get_pyfv3_gridbounds_a(FV_Atm(1)%bd)
            
            grid_handle = get_fv_grid_type_handle(FV_Atm(1)%gridstruct)
            flags_handle = get_fv_flags_type_handle(FV_Atm(1)%flagstruct)
            nest_handle = get_fv_nest_type_handle(FV_Atm(1)%neststruct)
            idiag_handle = get_fv_diag_type_handle(FV_Atm(1)%idiag)
            parent_grid_handle = get_fv_atmos_type_handle(FV_Atm(1)%parent_grid)
            domain_handle = get_domain2d_handle(FV_Atm(1)%domain)

            call pyfv3lm_fv_dynamics_r8(FV_Atm(1)%npx, FV_Atm(1)%npy, FV_Atm(1)%npz, FV_Atm(1)%ncnst, FV_Atm(1)%ng,  &
                                        real(conf%dt, fvprec), FV_Atm(1)%flagstruct%consv_te, c_fill,           &
                                        c_reproduce_sum, real(kappa, fvprec),                                   &
                                        real(cp, fvprec), real(zvir, fvprec), FV_Atm(1)%ptop, FV_Atm(1)%ks, FV_Atm(1)%flagstruct%ncnst,          &
                                        FV_Atm(1)%flagstruct%n_split, FV_Atm(1)%flagstruct%q_split,                  &
                                        FV_Atm(1)%u, FV_Atm(1)%v, FV_Atm(1)%w, FV_Atm(1)%delz,                       &
                                        c_hydrostatic, FV_Atm(1)%pt, FV_Atm(1)%delp, FV_Atm(1)%q, &
                                        FV_Atm(1)%ps, FV_Atm(1)%pe, FV_Atm(1)%pk, FV_Atm(1)%peln, FV_Atm(1)%pkz,     &
                                        FV_Atm(1)%phis, FV_Atm(1)%q_con, FV_Atm(1)%omga,                             &
                                        FV_Atm(1)%ua, FV_Atm(1)%va, FV_Atm(1)%uc, FV_Atm(1)%vc,                      &
                                        FV_Atm(1)%ak, FV_Atm(1)%bk,                                                  &
                                        FV_Atm(1)%mfx, FV_Atm(1)%mfy, FV_Atm(1)%cx, FV_Atm(1)%cy, FV_Atm(1)%ze0,     &
                                        c_hybrid_z, grid_handle, flags_handle,   &
                                        nest_handle, idiag_handle, c_bd, parent_grid_handle,  &
                                        domain_handle, dummy_time)

        else
            call fv_dynamics_nlm( FV_Atm(1)%npx, FV_Atm(1)%npy, FV_Atm(1)%npz, FV_Atm(1)%ncnst, FV_Atm(1)%ng,  &
                                  real(conf%dt, fvprec), FV_Atm(1)%flagstruct%consv_te, FV_Atm(1)%flagstruct%fill,           &
                                  FV_Atm(1)%flagstruct%reproduce_sum, real(kappa, fvprec),                                   &
                                  real(cp, fvprec), real(zvir, fvprec), FV_Atm(1)%ptop, FV_Atm(1)%ks, FV_Atm(1)%flagstruct%ncnst,          &
                                  FV_Atm(1)%flagstruct%n_split, FV_Atm(1)%flagstruct%q_split,                  &
                                  FV_Atm(1)%u, FV_Atm(1)%v, FV_Atm(1)%w, FV_Atm(1)%delz,                       &
                                  FV_Atm(1)%flagstruct%hydrostatic, FV_Atm(1)%pt, FV_Atm(1)%delp, FV_Atm(1)%q, &
                                  FV_Atm(1)%ps, FV_Atm(1)%pe, FV_Atm(1)%pk, FV_Atm(1)%peln, FV_Atm(1)%pkz,     &
                                  FV_Atm(1)%phis, FV_Atm(1)%q_con, FV_Atm(1)%omga,                             &
                                  FV_Atm(1)%ua, FV_Atm(1)%va, FV_Atm(1)%uc, FV_Atm(1)%vc,                      &
                                  FV_Atm(1)%ak, FV_Atm(1)%bk,                                                  &
                                  FV_Atm(1)%mfx, FV_Atm(1)%mfy, FV_Atm(1)%cx, FV_Atm(1)%cy, FV_Atm(1)%ze0,     &
                                  FV_Atm(1)%flagstruct%hybrid_z, FV_Atm(1)%gridstruct, FV_Atm(1)%flagstruct,   &
                                  self%FV_AtmP(1)%flagstruct,                                                       &
                                  FV_Atm(1)%neststruct, FV_Atm(1)%idiag, FV_Atm(1)%bd, FV_Atm(1)%parent_grid,  &
                                  FV_Atm(1)%domain )
        endif
       
       
        ! MPP nulify
        ! ----------
        call nullify_domain()
       
        !Copy from fv3 back to traj structure
        !------------------------------------
        call fv3_to_traj(self,conf,traj)

    end subroutine

    function get_fv_grid_type_handle(obj) result(handle)
        type fv_grid_type_ptr
            type(fv_grid_type), pointer :: p => NULL()
        end type
        type(fv_grid_type), intent(in), target :: obj
        type(fv_grid_type_ptr) :: obj_ptr
        integer(c_int), dimension(2) :: handle
        obj_ptr%p => obj
        handle = transfer(obj_ptr, handle)
    end function

    function get_fv_flags_type_handle(obj) result(handle)
        type fv_flags_type_ptr
            type(fv_flags_type), pointer :: p => NULL()
        end type
        type(fv_flags_type), intent(in), target :: obj
        type(fv_flags_type_ptr) :: obj_ptr
        integer(c_int), dimension(2) :: handle
        obj_ptr%p => obj
        handle = transfer(obj_ptr, handle)
    end function

    function get_fv_nest_type_handle(obj) result(handle)
        type fv_nest_type_ptr
            type(fv_nest_type), pointer :: p => NULL()
        end type
        type(fv_nest_type), intent(in), target :: obj
        type(fv_nest_type_ptr) :: obj_ptr
        integer(c_int), dimension(2) :: handle
        obj_ptr%p => obj
        handle = transfer(obj_ptr, handle)
    end function

    function get_fv_atmos_type_handle(obj) result(handle)
        type fv_atmos_type_ptr
            type(fv_atmos_type), pointer :: p => NULL()
        end type
        type(fv_atmos_type), intent(in), target :: obj
        type(fv_atmos_type_ptr) :: obj_ptr
        integer(c_int), dimension(2) :: handle
        obj_ptr%p => obj
        handle = transfer(obj_ptr, handle)
    end function

    function get_fv_diag_type_handle(obj) result(handle)
        type fv_diag_type_ptr
            type(fv_diag_type), pointer :: p => NULL()
        end type
        type(fv_diag_type), intent(in), target :: obj
        type(fv_diag_type_ptr) :: obj_ptr
        integer(c_int), dimension(2) :: handle
        obj_ptr%p => obj
        handle = transfer(obj_ptr, handle)
    end function

    function get_domain2d_handle(obj) result(handle)
        type domain2d_ptr
            type(domain2d), pointer :: p => NULL()
        end type
        type(domain2d), intent(in), target :: obj
        type(domain2d_ptr) :: obj_ptr
        integer(c_int), dimension(2) :: handle
        obj_ptr%p => obj
        handle = transfer(obj_ptr, handle)
    end function

    function get_pyfv3_gridbounds_a(bd) result(c_bd)
        type(fv_grid_bounds_type), intent(in) :: bd
        type(PyFV3LM_GridBounds_A) :: c_bd
        c_bd%Is = bd%is
        c_bd%Ie = bd%ie
        c_bd%Js = bd%js
        c_bd%Je = bd%je
        c_bd%Isd = bd%isd
        c_bd%Ied = bd%ied
        c_bd%Jsd = bd%jsd
        c_bd%Jed = bd%jed
        c_bd%Isc = bd%isc
        c_bd%Iec = bd%iec
        c_bd%Jsc = bd%jsc
        c_bd%Jec = bd%jec
        c_bd%ng = bd%ng
    end function

    subroutine traj_to_fv3(self,conf,traj)

        implicit none
       
        class(fv3jedi_lm_dynamics_type), intent(inout) :: self
        type(fv3jedi_lm_conf), intent(in) :: conf
        type(fv3jedi_lm_traj), intent(in) :: traj
       
        integer :: i,j,k
       
        !Zero the halos
        !--------------
        self%FV_Atm(1)%u     = 0.0_kind_real
        self%FV_Atm(1)%v     = 0.0_kind_real
        self%FV_Atm(1)%pt    = 0.0_kind_real
        self%FV_Atm(1)%delp  = 0.0_kind_real
        self%FV_Atm(1)%q     = 0.0_kind_real
        self%FV_Atm(1)%w     = 0.0_kind_real
        self%FV_Atm(1)%delz  = 0.0_kind_real
        self%FV_Atm(1)%phis  = 0.0_kind_real
        self%FV_Atm(1)%pe    = 0.0_kind_real
        self%FV_Atm(1)%peln  = 0.0_kind_real
        self%FV_Atm(1)%pk    = 0.0_kind_real
        self%FV_Atm(1)%pkz   = 0.0_kind_real
        self%FV_Atm(1)%ua    = 0.0_kind_real
        self%FV_Atm(1)%va    = 0.0_kind_real
        self%FV_Atm(1)%uc    = 0.0_kind_real
        self%FV_Atm(1)%vc    = 0.0_kind_real
        self%FV_Atm(1)%omga  = 0.0_kind_real
        self%FV_Atm(1)%mfx   = 0.0_kind_real
        self%FV_Atm(1)%mfy   = 0.0_kind_real
        self%FV_Atm(1)%cx    = 0.0_kind_real
        self%FV_Atm(1)%cy    = 0.0_kind_real
        self%FV_Atm(1)%ze0   = 0.0_kind_real
        self%FV_Atm(1)%q_con = 0.0_kind_real
        self%FV_Atm(1)%ps    = 0.0_kind_real
       
       
        !Copy from traj
        !--------------
        self%FV_Atm(1)%u   (self%isc:self%iec,self%jsc:self%jec,:) = traj%u   (self%isc:self%iec,self%jsc:self%jec,:)
        self%FV_Atm(1)%v   (self%isc:self%iec,self%jsc:self%jec,:) = traj%v   (self%isc:self%iec,self%jsc:self%jec,:)
        self%FV_Atm(1)%pt  (self%isc:self%iec,self%jsc:self%jec,:) = traj%t   (self%isc:self%iec,self%jsc:self%jec,:)
        self%FV_Atm(1)%delp(self%isc:self%iec,self%jsc:self%jec,:) = traj%delp(self%isc:self%iec,self%jsc:self%jec,:)
        self%FV_Atm(1)%q(self%isc:self%iec,self%jsc:self%jec,:,:) = traj%tracers(self%isc:self%iec,self%jsc:self%jec,:,:)
       
        if (.not. self%FV_Atm(1)%flagstruct%hydrostatic) then
            self%FV_Atm(1)%delz(self%isc:self%iec  ,self%jsc:self%jec  ,:  ) = traj%delz(self%isc:self%iec  ,self%jsc:self%jec  ,:  )
            self%FV_Atm(1)%w   (self%isc:self%iec  ,self%jsc:self%jec  ,:  ) = traj%w   (self%isc:self%iec  ,self%jsc:self%jec  ,:  )
        endif
       
        self%FV_Atm(1)%phis(self%isc:self%iec,self%jsc:self%jec) = traj%phis(self%isc:self%iec,self%jsc:self%jec)
       
        !Update edges of d-grid winds
        !----------------------------
        call mpp_get_boundary(self%FV_Atm(1)%u, self%FV_Atm(1)%v, self%FV_Atm(1)%domain, &
                              wbuffery=self%wbuffery, ebuffery=self%ebuffery, &
                              sbufferx=self%sbufferx, nbufferx=self%nbufferx, &
                              gridtype=DGRID_NE, complete=.true. )
        do k=1,self%npz
            do i=self%isc,self%iec
                self%FV_Atm(1)%u(i,self%jec+1,k) = self%nbufferx(i,k)
            enddo
        enddo
        do k=1,self%npz
            do j=self%jsc,self%jec
                self%FV_Atm(1)%v(self%iec+1,j,k) = self%ebuffery(j,k)
            enddo
        enddo
       
       
        ! Fill phi halos
        ! --------------
        call mpp_update_domains(self%FV_Atm(1)%phis, self%FV_Atm(1)%domain, complete=.true.)
       
       
        !Compute the other pressure variables needed by FV3
        !--------------------------------------------------
        call compute_fv3_pressures( self%isc, self%iec, self%jsc, self%jec, self%isd, self%ied, self%jsd, self%jed, &
                                    self%npz, real(kappa, fvprec), self%FV_Atm(1)%ptop, &
                                    self%FV_Atm(1)%delp, self%FV_Atm(1)%pe, self%FV_Atm(1)%pk, self%FV_Atm(1)%pkz, self%FV_Atm(1)%peln )
       
    end subroutine traj_to_fv3
       
    subroutine fv3_to_traj(self,conf,traj)
       
        implicit none
       
        class(fv3jedi_lm_dynamics_type), intent(in) :: self
        type(fv3jedi_lm_conf), intent(in)    :: conf
        type(fv3jedi_lm_traj), intent(inout) :: traj
       
        traj%u   (self%isc:self%iec,self%jsc:self%jec,:) = self%FV_Atm(1)%u   (self%isc:self%iec,self%jsc:self%jec,:)
        traj%v   (self%isc:self%iec,self%jsc:self%jec,:) = self%FV_Atm(1)%v   (self%isc:self%iec,self%jsc:self%jec,:)
        traj%t   (self%isc:self%iec,self%jsc:self%jec,:) = self%FV_Atm(1)%pt  (self%isc:self%iec,self%jsc:self%jec,:)
        traj%delp(self%isc:self%iec,self%jsc:self%jec,:) = self%FV_Atm(1)%delp(self%isc:self%iec,self%jsc:self%jec,:)
        traj%tracers(self%isc:self%iec,self%jsc:self%jec,:,:) = self%FV_Atm(1)%q(self%isc:self%iec,self%jsc:self%jec,:,:)
       
        if (.not. self%FV_Atm(1)%flagstruct%hydrostatic) then
            traj%delz(self%isc:self%iec,self%jsc:self%jec,:) = self%FV_Atm(1)%delz(self%isc:self%iec,self%jsc:self%jec,:)
            traj%w   (self%isc:self%iec,self%jsc:self%jec,:) = self%FV_Atm(1)%w   (self%isc:self%iec,self%jsc:self%jec,:)
        endif
       
        traj%ua(self%isc:self%iec,self%jsc:self%jec,:) = self%FV_Atm(1)%ua(self%isc:self%iec,self%jsc:self%jec,:)
        traj%va(self%isc:self%iec,self%jsc:self%jec,:) = self%FV_Atm(1)%va(self%isc:self%iec,self%jsc:self%jec,:)
       
        traj%phis(self%isc:self%iec,self%jsc:self%jec) = self%FV_Atm(1)%phis(self%isc:self%iec,self%jsc:self%jec)
       
    end subroutine fv3_to_traj
    
end module pyfv3lm_fv3jedilm_mock_mod