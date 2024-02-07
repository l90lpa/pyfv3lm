module mpp_ops
    use iso_c_binding
    use mpp_mod, only: mpp_init, mpp_exit, mpp_sync, mpp_set_root_pe, mpp_pe, mpp_npes
    use mpp_domains_mod, only: mpp_domains_init, mpp_domains_set_stack_size, mpp_domains_exit, &
                               mpp_define_layout, mpp_define_domains, mpp_update_domains, &
                               mpp_update_domains_ad, mpp_get_compute_domain, mpp_get_data_domain
    use interface_utils, only: domain2d_ptr, handle_to_ptr_domain2d, ptr_to_handle_domain2d, int_to_logical
    implicit none
    
contains

    subroutine py_mpp_init() bind(c, name="py_mpp_init")
        implicit none
        call mpp_init()
    end subroutine


    subroutine py_mpp_pe(pe) bind(c, name="py_mpp_pe")
        implicit none
        integer(c_int), intent(inout) :: pe
        pe = mpp_pe()
    end subroutine


    subroutine py_mpp_npes(npes) bind(c, name="py_mpp_npes")
        implicit none
        integer(c_int), intent(inout) :: npes
        npes = mpp_npes()
    end subroutine


    subroutine py_mpp_set_root_pe(root) bind(c, name="py_mpp_set_root_pe")
        implicit none
        integer(c_int), intent(in) :: root
        call mpp_set_root_pe(root)
    end subroutine


    subroutine py_mpp_exit() bind(c, name="py_mpp_exit")
        implicit none
        call mpp_exit()
    end subroutine


    subroutine py_mpp_sync() bind(c, name="py_mpp_sync")
        implicit none
        call mpp_sync()
    end subroutine

    
    subroutine py_mpp_domains_init() bind(c, name="py_mpp_domains_init")
        implicit none
        call mpp_domains_init()
    end subroutine
    

    subroutine py_mpp_domains_set_stack_size(stack_size) bind(c, name="py_mpp_domains_set_stack_size")
        implicit none
        integer(c_int), intent(in) :: stack_size
        call mpp_domains_set_stack_size(stack_size)
    end subroutine


    subroutine py_mpp_domains_exit() bind(c, name="py_mpp_domains_exit")
        implicit none
        call mpp_domains_exit()
    end subroutine
    
    
    subroutine py_mpp_allocate_domain2D(domain_handle) bind(c, name="py_mpp_allocate_domain2D")
        implicit none
        integer(c_int), intent(inout) :: domain_handle(2) !< 2D domain decomposition to define
        
        type(domain2d_ptr) :: domain_ptr
        
        allocate(domain_ptr%data)
        domain_handle = ptr_to_handle_domain2d(domain_ptr)
        
    end subroutine
    
    
    subroutine py_mpp_deallocate_domain2D(domain_handle) &
        bind(c, name="py_mpp_deallocate_domain2D")
        implicit none
        integer(c_int), intent(inout) :: domain_handle(2) !< 2D domain decomposition to define

        type(domain2d_ptr) :: domain_ptr

        domain_ptr = handle_to_ptr_domain2d(domain_handle)
        deallocate(domain_ptr%data)

    end subroutine


    subroutine py_mpp_get_compute_domain2d(domain_handle, compute_domain_indices) &
        bind(c, name="py_mpp_get_compute_domain2d")
        implicit none
        integer(c_int), intent(in) :: domain_handle(2)
        integer(c_int), intent(inout) :: compute_domain_indices(4)

        type(domain2d_ptr) :: domain_ptr
        integer(c_int) :: xstart, xend, ystart, yend

        domain_ptr = handle_to_ptr_domain2d(domain_handle)
        call mpp_get_compute_domain(domain_ptr%data, xstart, xend, ystart, yend)
        compute_domain_indices(1) = xstart
        compute_domain_indices(2) = xend
        compute_domain_indices(3) = ystart
        compute_domain_indices(4) = yend
    end subroutine


    subroutine py_mpp_get_data_domain2d(domain_handle, data_domain_indices) &
        bind(c, name="py_mpp_get_data_domain2d")
        implicit none
        integer(c_int), intent(in) :: domain_handle(2)
        integer(c_int), intent(inout) :: data_domain_indices(4)

        type(domain2d_ptr) :: domain_ptr
        integer(c_int) :: xstart, xend, ystart, yend

        domain_ptr = handle_to_ptr_domain2d(domain_handle)
        call mpp_get_data_domain(domain_ptr%data, xstart, xend, ystart, yend)
        data_domain_indices(1) = xstart
        data_domain_indices(2) = xend
        data_domain_indices(3) = ystart
        data_domain_indices(4) = yend
    end subroutine


    subroutine py_mpp_define_layout2D( global_indices, n_indices, ndivs, layout, n_layout ) &
        bind(c, name="py_mpp_define_layout2D")
        implicit none
        integer(c_int), intent(in) :: global_indices(n_indices) !< (/ isg, ieg, jsg, jeg /); Defines the global domain.
        !f2py intent(hide) :: n_indices
        integer(c_int), intent(in) :: n_indices
        integer(c_int), intent(in) :: ndivs                            !< number of divisions to divide global domain
        integer(c_int), intent(inout) :: layout(n_layout)
        !f2py intent(hide) :: n_layout
        integer(c_int), intent(in) :: n_layout

        call mpp_define_layout(global_indices, ndivs, layout)
    
    end subroutine


    subroutine py_mpp_define_domains2D(global_indices, n_indices, layout, n_layout, domain_handle, &
                                       xhalo, yhalo, print_info) &
        bind(c, name="py_mpp_define_domains2D")
        implicit none

        integer(c_int), intent(in)  :: global_indices(n_indices) !<(/ isg, ieg, jsg, jeg /)
        !f2py intent(hide) :: n_indices
        integer(c_int), intent(in) :: n_indices
        integer(c_int), intent(in)  :: layout(n_layout) !< pe layout
        !f2py intent(hide) :: n_layout
        integer(c_int), intent(in) :: n_layout
        integer(c_int), intent(in)  :: domain_handle(2) !< 2D domain decomposition to define
        integer(c_int), intent(in)  :: xhalo, yhalo !< halo sizes for x and y indices
        integer(c_int), intent(in)  :: print_info

        type(domain2d_ptr) :: domain_ptr

        domain_ptr = handle_to_ptr_domain2d(domain_handle)

        if (print_info == 0) then
            call mpp_define_domains(global_indices, layout, domain_ptr%data, xhalo=xhalo, yhalo=yhalo)
        else
            call mpp_define_domains(global_indices, layout, domain_ptr%data, xhalo=xhalo, yhalo=yhalo, name="un-named")
        end if
    
    end subroutine


    !updates data domain of 3D field whose computational domains have been computed
    subroutine py_mpp_update_domain2D_r4_3dv(fieldx, xi, xj, xk, fieldy, yi, yj, yk, &
                                             domain_handle, grid_type) & 
        bind(c, name="py_mpp_update_domain2D_r4_3dv")
        implicit none

        real(c_float),   intent(inout) :: fieldx(xi,xj,xk), fieldy(yi,yj,yk)
        !f2py intent(hide) :: xi, xj, xk, yi, yj, yk
        integer(c_int),   intent(in)    :: xi, xj, xk, yi, yj, yk
        integer(c_int),   intent(in)    :: domain_handle(2)
        integer(c_int),   intent(in)    :: grid_type

        type(domain2d_ptr) :: domain_ptr

        domain_ptr = handle_to_ptr_domain2d(domain_handle)

        call mpp_update_domains(fieldx, fieldy, domain_ptr%data, gridtype=grid_type)

    end subroutine


    !adjoint of py_mpp_update_domain2D_r4_3dv
    subroutine py_mpp_update_domain2D_r4_3dv_ad(fieldx, xi, xj, xk, fieldy, yi, yj, yk, &
                                                domain_handle, grid_type) & 
        bind(c, name="py_mpp_update_domain2D_r4_3dv_ad")
        implicit none

        real(c_float),   intent(inout) :: fieldx(xi,xj,xk), fieldy(yi,yj,yk)
        !f2py intent(hide) :: xi, xj, xk, yi, yj, yk
        integer(c_int),   intent(in)    :: xi, xj, xk, yi, yj, yk
        integer(c_int),   intent(in)    :: domain_handle(2)
        integer(c_int),   intent(in)    :: grid_type

        type(domain2d_ptr) :: domain_ptr

        domain_ptr = handle_to_ptr_domain2d(domain_handle)

        call mpp_update_domains_ad(fieldx, fieldy, domain_ptr%data, gridtype=grid_type)

    end subroutine


    !updates data domain of 3D field whose computational domains have been computed
    subroutine py_mpp_update_domain2D_r8_3dv(fieldx, xi, xj, xk, fieldy, yi, yj, yk, &
                                             domain_handle, grid_type) & 
        bind(c, name="py_mpp_update_domain2D_r8_3dv")
        implicit none

        real(c_double),   intent(inout) :: fieldx(xi,xj,xk), fieldy(yi,yj,yk)
        !f2py intent(hide) :: xi, xj, xk, yi, yj, yk
        integer(c_int),   intent(in)    :: xi, xj, xk, yi, yj, yk
        integer(c_int),   intent(in)    :: domain_handle(2)
        integer(c_int),   intent(in)    :: grid_type
        
        type(domain2d_ptr) :: domain_ptr
        
        domain_ptr = handle_to_ptr_domain2d(domain_handle)

        call mpp_update_domains(fieldx, fieldy, domain_ptr%data, gridtype=grid_type)

    end subroutine


    !adjoint of py_mpp_update_domain2D_r8_3dv
    subroutine py_mpp_update_domain2D_r8_3dv_ad(fieldx, xi, xj, xk, fieldy, yi, yj, yk, &
                                                domain_handle, grid_type) & 
        bind(c, name="py_mpp_update_domain2D_r8_3dv_ad")
        implicit none

        real(c_double),   intent(inout) :: fieldx(xi,xj,xk), fieldy(yi,yj,yk)
        !f2py intent(hide) :: xi, xj, xk, yi, yj, yk
        integer(c_int),   intent(in)    :: xi, xj, xk, yi, yj, yk
        integer(c_int),   intent(in)    :: domain_handle(2)
        integer(c_int),   intent(in)    :: grid_type

        type(domain2d_ptr) :: domain_ptr

        domain_ptr = handle_to_ptr_domain2d(domain_handle)

        call mpp_update_domains_ad(fieldx, fieldy, domain_ptr%data, gridtype=grid_type)

    end subroutine
    
end module