module omars_data_matrix
    !$ use omp_lib
    implicit none
    !f2py threadsafef
contains
    subroutine data_matrix(x, basis_slice_start, basis_slice_end, covariates, nodes, hinges, where, result)
        ! Declare inputs
        real(8), intent(in) :: x(:, :)
        integer, intent(in) :: basis_slice_start, basis_slice_end
        integer, intent(in) :: covariates(:, :)
        real(8), intent(in) :: nodes(:, :)
        real(8), intent(in) :: hinges(:, :)
        integer, intent(in) :: where(:, :)

        ! Declare output
        real(8), intent(out) :: result(size(x, 1), basis_slice_end - basis_slice_start)

        ! Declare local variables
        integer :: basis_idx, func_idx
        real(8) :: temp(size(x, 1))

        ! Initialize result
        result = 1.0d0

        ! Perform the calculation
        !$OMP PARALLEL DO PRIVATE(basis_idx, func_idx, temp)
        do basis_idx = basis_slice_start + 1, basis_slice_end
            do func_idx = 1, size(nodes, 1)
                if (where(func_idx, basis_idx) == 1) then
                    temp = x(:, covariates(func_idx, basis_idx) + 1) - nodes(func_idx, basis_idx)
                    if (hinges(func_idx, basis_idx) == 1) then
                        temp = max(0.0d0, temp)
                    end if
                    result(:, basis_idx) = result(:, basis_idx) * temp
                end if
            end do
        end do
        !$OMP END PARALLEL DO
    end subroutine data_matrix
end module omars_data_matrix
