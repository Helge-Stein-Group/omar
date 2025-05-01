! filepath: /home/tim/projects/omar/omar/tests/unit_test.f90
program fortran_unit_test
    use backend
    use utils
    implicit none

    ! Variables for tests
    real(8) :: x(N_SAMPLES, DIM), y(N_SAMPLES), y_true(N_SAMPLES)
    integer :: nbases
    logical :: mask(5, 5), truncated(5, 5)
    integer :: cov(5, 5)
    real(8) :: root(5, 5)
    real(8) :: y_mean
    integer :: penalty

    ! Reference calculation variables
    integer :: ref_indices(4)
    real(8) :: ref_data_matrix(N_SAMPLES, 4), ref_data_matrix_mean(4)
    real(8) :: ref_cov_matrix(4, 4)
    real(8) :: ref_rhs(4)
    real(8) :: ref_coefficients(4), ref_chol(4, 4)
    real(8) :: ref_lof

    ! Backend output variables
    integer :: indices(4)
    real(8) :: data_matrix_out(N_SAMPLES, 4), data_matrix_mean_out(4)
    real(8) :: cov_matrix_out(4, 4)
    real(8) :: rhs_out(4)
    real(8) :: coefficients_out(4), chol_out(4, 4)
    real(8) :: lof_out

    ! Variables for update test
    real(8), allocatable :: data_matrix_update(:,:), data_matrix_mean_update(:)
    real(8), allocatable :: cov_matrix_update(:,:), rhs_update(:), chol_update(:,:)
    real(8), allocatable :: coefficients_update(:)
    real(8) :: prev_root_val
    real(8) :: next_roots(3)
    integer :: i, k, n, info
    integer :: parent_idx_update
    real(8) :: lof_update

    ! Variables for expand/prune/find tests
    integer, parameter :: max_nbases_test = 11 ! Example value
    integer :: nbases_expand, nbases_prune, nbases_find
    logical :: mask_expand(max_nbases_test, max_nbases_test)
    logical :: truncated_expand(max_nbases_test, max_nbases_test)
    integer :: cov_expand(max_nbases_test, max_nbases_test)
    real(8) :: root_expand(max_nbases_test, max_nbases_test)
    real(8) :: coefficients_expand(max_nbases_test - 1)
    real(8) :: lof_expand
    logical :: mask_prune(max_nbases_test, max_nbases_test)
    real(8) :: coefficients_prune(max_nbases_test - 1)
    real(8) :: lof_prune
    logical :: mask_find(max_nbases_test, max_nbases_test)
    logical :: truncated_find(max_nbases_test, max_nbases_test)
    integer :: cov_find(max_nbases_test, max_nbases_test)
    real(8) :: root_find(max_nbases_test, max_nbases_test)
    real(8) :: coefficients_find(max_nbases_test - 1)
    real(8) :: lof_find
    integer :: max_ncandidates_test = 10
    real(8) :: aging_factor_test = 0.1

    ! --- Setup ---
    penalty = 3
    print *, "--- Running Fortran Backend Tests ---"
    call generate_data(x, y, y_true)
    call reference_model(x, nbases, mask, truncated, cov, root)
    y_mean = sum(y) / N_SAMPLES
    
    call reference_data_matrix(x, ref_data_matrix, ref_data_matrix_mean)
    call reference_covariance_matrix(ref_data_matrix, ref_cov_matrix)
    call reference_rhs(y, ref_data_matrix, ref_rhs)
    ! Reference calculation requires LAPACK calls (assuming available)
    ref_chol = ref_cov_matrix
    call dpotrf('L', size(ref_chol, 1), ref_chol, size(ref_chol, 1), info)
    if (info /= 0) then
        print *, "Error during Cholesky decomposition, info = ", info
        stop
    end if
    do i = 1, 4
        ref_chol(i, i+1:4) = 0.0d0
    end do
    ref_coefficients = ref_rhs
    call dpotrs('L', size(ref_chol, 1), 1, ref_chol, size(ref_chol, 1), &
                ref_coefficients, size(ref_chol, 1), info)
    if (info /= 0) then
        print *, "Error during solving linear system with dpotrs, info = ", info
        stop
    end if
    

    ! --- Run Tests ---
    call test_active_base_indices()
    call test_data_matrix()
    call test_covariance_matrix()
    call test_rhs()
    call test_coefficients()
    call test_fit()
    ! call test_update_fortran() ! Complex test, might need refinement
    ! call test_expand_bases_fortran() ! Verification might be basic
    ! call test_prune_bases_fortran() ! Verification might be basic
    ! call test_find_bases_fortran() ! Verification might be basic

contains

    subroutine test_active_base_indices()
        print *, " "
        print *, "--- Testing active_base_indices ---"
        call active_base_indices(mask, nbases, indices)
        ref_indices = [2, 3, 4, 5]
        call compare_int_arrays(indices, ref_indices, "Active Base Indices")
    end subroutine test_active_base_indices

    subroutine test_data_matrix()
        print *, " "
        print *, "--- Testing data_matrix ---"
        call active_base_indices(mask, nbases, indices)
        call data_matrix(x, indices, mask, truncated, cov, root, data_matrix_out, &
        data_matrix_mean_out)

        call compare_real_matrices(data_matrix_out, ref_data_matrix, "Data Matrix")
        call compare_real_arrays(data_matrix_mean_out, ref_data_matrix_mean, "Data Matrix Mean")
    end subroutine test_data_matrix

    subroutine test_covariance_matrix()
        print *, " "
        print *, "--- Testing covariance_matrix ---"
        call covariance_matrix(ref_data_matrix, cov_matrix_out)
        call compare_real_matrices(cov_matrix_out, ref_cov_matrix, "Covariance Matrix")
    end subroutine test_covariance_matrix

    subroutine test_rhs()
        print *, " "
        print *, "--- Testing rhs ---"
        call rhs(y, y_mean, ref_data_matrix, rhs_out)
        call compare_real_arrays(rhs_out, ref_rhs, "RHS")
    end subroutine test_rhs

    subroutine test_coefficients()
        print *, " "
        print *, "--- Testing coefficients ---"
        call coefficients(ref_cov_matrix, ref_rhs, coefficients_out, chol_out)
        do i = 1, size(chol_out, 1)
            chol_out(i, i+1:) = 0.0d0
        end do
        
        call compare_real_matrices(chol_out, ref_chol, "Cholesky Factor")
        call compare_real_arrays(coefficients_out, ref_coefficients, "Coefficients")        
    end subroutine test_coefficients

    subroutine test_fit()
        print *, " "
        print *, "--- Testing fit ---"
        call fit(x, y, y_mean, nbases, mask, truncated, cov, root, penalty, &
                 data_matrix_out, data_matrix_mean_out, cov_matrix_out, rhs_out, chol_out, coefficients_out, lof_out)
        do i = 1, size(chol_out, 1)
            chol_out(i, i+1:) = 0.0d0
        end do

        call compare_real_matrices(data_matrix_out, ref_data_matrix, "Fit: Data Mat")
        call compare_real_arrays(data_matrix_mean_out, ref_data_matrix_mean, "Fit: Data Mean")
        call compare_real_matrices(cov_matrix_out, ref_cov_matrix, "Fit: Cov Mat")
        call compare_real_arrays(rhs_out, ref_rhs, "Fit: RHS")
        call compare_real_matrices(chol_out, ref_chol, "Fit: Chol")
        call compare_real_arrays(coefficients_out, ref_coefficients, "Fit: Coeffs")
    end subroutine test_fit

    ! subroutine test_update_fortran()
    !     ! This test is significantly more complex to translate accurately,
    !     ! especially the comparison logic. Skipping for now.
    !     print *, " "
    !     print *, "--- Testing update_fit (SKIPPED) ---"
    ! end subroutine test_update_fortran

    ! subroutine test_expand_bases_fortran()
    !     ! Verification requires comparing against a known complex result or
    !     ! analyzing the output structure. Skipping detailed verification for now.
    !     print *, " "
    !     print *, "--- Testing expand_bases (Basic Run) ---"
    !     call expand_bases(x, y, y_mean, max_nbases_test, max_ncandidates_test, aging_factor_test, penalty, &
    !                       lof_expand, nbases_expand, mask_expand, truncated_expand, cov_expand, root_expand, &
    !                       coefficients_expand)
    !     print *, "Expand Bases Result: nbases =", nbases_expand, "LOF =", lof_expand
    !     print *, "Expand Bases: PASSED (if no crash)"
    ! end subroutine test_expand_bases_fortran

    ! subroutine test_prune_bases_fortran()
    !     ! Requires setting up a larger initial state and comparing the pruned
    !     ! state to the original reference model. Skipping detailed verification.
    !     print *, " "
    !     print *, "--- Testing prune_bases (SKIPPED) ---"
    ! end subroutine test_prune_bases_fortran

    ! subroutine test_find_bases_fortran()
    !     ! Calls expand and prune. Verification is complex.
    !     print *, " "
    !     print *, "--- Testing find_bases (Basic Run) ---"
    !      call find_bases(x, y, y_mean, max_nbases_test, max_ncandidates_test, aging_factor_test, penalty, &
    !                      lof_find, nbases_find, mask_find, truncated_find, cov_find, root_find, &
    !                      coefficients_find)
    !     print *, "Find Bases Result: nbases =", nbases_find, "LOF =", lof_find
    !     print *, "Find Bases: PASSED (if no crash)"
    ! end subroutine test_find_bases_fortran


end program fortran_unit_test