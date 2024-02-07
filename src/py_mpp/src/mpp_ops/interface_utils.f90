module interface_utils
    use iso_c_binding
    use mpp_domains_mod, only: domain2d
    implicit none

    type :: domain2d_ptr
        type(domain2d), pointer :: data => NULL()
    end type
    
contains

    function handle_to_ptr_domain2d(handle) result(ptr)
        integer(c_int), intent(in) :: handle(2)
        type(domain2d_ptr) :: ptr
        ptr = transfer(handle, ptr)
    end function

    function ptr_to_handle_domain2d(ptr) result(handle)
        type(domain2d_ptr), intent(in) :: ptr
        integer(c_int) :: handle(2)
        handle = transfer(ptr, handle)
    end function

    function int_to_logical(int) result(bool)
        integer(c_int), intent(in) :: int
        
        logical :: bool

        if (int == 0)  then
            bool = .false.
        else
            bool = .true.
        end if
    end function
    
end module