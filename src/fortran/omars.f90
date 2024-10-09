module omars
    !$ use omp_lib
    implicit none
    !f2py threadsafef
contains
    subroutine update_cholesky(chol, update_vectors, multipliers)
        ! Declar inputs
        real(8), intent(in) :: update_vectors(:, :)
        real(8), intent(in) :: multipliers(2)
        real(8), intent(inout) :: chol(:, :)

        ! Declare local variables
        real(8) :: diag(size(chol, 1))
        real(8) :: u(size(chol, 1), size(update_vectors, 1))
        real(8) :: b(size(chol, 1))
        integer :: i, update_idx

        do update_idx = 1, 2
            ! Extract diagonal
            do i = 1, size(chol, 1)
                diag(i) = chol(i, i)
            end do
            ! Divide chol by its diagonal elements (broadcast manually)
            do i = 1, size(chol, 1)
                chol(i, :) = chol(i, :) / diag(i)
            end do
            diag = diag**2

            u = 0.0
            u(:, 1) = update_vectors(:, update_idx)
            u(2:, 1) = u(2:, 1) - update_vectors(1, update_idx) * chol(1, 2:)
            b = 1.0

            do i = 2, size(chol, 1)
                u(:, i) = u(:, i - 1)
                u(i + 1:, i) = u(i + 1:, i) - u(i, i - 1) * chol(i, i + 1:)
                b(i) = b(i - 1) + multipliers(update_idx) * u(i - 1, i - 1)**2 / diag(i - 1)
            end do

            do i = 1, size(chol, 1)
                chol(i, i) = sqrt(diag(i) + multipliers(update_idx) / b(i) * u(i, i)**2)
                chol(i, i + 1:) = chol(i, i + 1:) * chol(i, i)
                chol(i, i + 1:) = chol(i, i + 1:) + multipliers(update_idx) / b(i) * u(i, i) * u(i + 1:, i) / chol(i, i)
            end do
        end do
    end subroutine update_cholesky

    subroutine decompose_addition(covariance_addition, eigenvalues, eigenvectors)
        ! Declare inputs
        real(8), intent(in) :: covariance_addition(:)

        ! Declar outputs
        real(8), intent(out) :: eigenvalues(2)
        real(8), intent(out) :: eigenvectors(size(covariance_addition), 2)

        ! Declare local variables
        real(8) :: eigenvalue_intermediate
        integer :: n, i

        n = size(covariance_addition)

        eigenvalue_intermediate = sqrt(covariance_addition(n)**2 + 4 * sum(covariance_addition(1:n - 1)**2))

        eigenvalues(1) = (covariance_addition(n) + eigenvalue_intermediate) / 2
        eigenvalues(2) = (covariance_addition(n) - eigenvalue_intermediate) / 2

        do i = 1, 2
            eigenvectors(1:n - 1, i) = covariance_addition(1:n - 1) / eigenvalues(i)
            eigenvectors(n, i) = 1.0d0
            eigenvectors(:, i) = eigenvectors(:, i) / sqrt(sum(eigenvectors(:, i)**2))
        end do

    end subroutine decompose_addition

    subroutine active_base_indices(where, nbases, result)
        ! Declare inputs
        integer, intent(in) :: nbases
        logical, intent(in) :: where(:, :)

        ! Declare output
        integer, intent(out) :: result(nbases)

        ! Declare local variables
        integer :: j, n_cols
        logical :: col_has_true(size(where, 2))

        n_cols = size(where, 2)

        col_has_true = any(where, dim = 1)

        result = pack([(j, j = 1, n_cols)], col_has_true)

    end subroutine active_base_indices

    subroutine data_matrix(x, basis_indices, covariates, nodes, hinges, where, result, result_mean)
        ! Declare inputs
        real(8), intent(in) :: x(:, :)
        integer, intent(in) :: basis_indices(:)
        integer, intent(in) :: covariates(:, :)
        real(8), intent(in) :: nodes(:, :)
        logical, intent(in) :: hinges(:, :)
        logical, intent(in) :: where(:, :)

        ! Declare output
        real(8), intent(out) :: result(size(x, 1), size(basis_indices))
        real(8), intent(out) :: result_mean(size(basis_indices))

        ! Declare local variables
        integer :: i, basis_idx, func_idx
        real(8) :: intermediate_result(size(x, 1))

        ! Initialize result
        result = 1.0d0

        ! Perform the calculation
        !$OMP PARALLEL DO PRIVATE(basis_idx, func_idx, temp)
        do i = 1, size(basis_indices)
            basis_idx = basis_indices(i)
            do func_idx = 1, size(where, 1)
                if (where(func_idx, basis_idx)) then
                    intermediate_result = x(:, covariates(func_idx, basis_idx) + 1) - nodes(func_idx, basis_idx)
                    if (hinges(func_idx, basis_idx)) then
                        intermediate_result = max(0.0d0, intermediate_result)
                    end if
                    result(:, i) = result(:, i) * intermediate_result
                end if
            end do
            result_mean(i) = sum(result(:, i)) / size(x, 1)
            result(:, i) = result(:, i) - result_mean(i)
        end do
        !$OMP END PARALLEL DO
    end subroutine data_matrix

    subroutine update_init(x, old_node, parent_idx, nbases, covariates, nodes, where, fit_matrix, basis_mean, update, &
            update_mean)
        ! Declare inputs
        real(8), intent(in) :: x(:, :)
        real(8), intent(in) :: old_node
        integer, intent(in) :: parent_idx
        integer, intent(in) :: nbases
        integer, intent(in) :: covariates(:, :)
        real(8), intent(in) :: nodes(:, :)
        logical, intent(in) :: where(:, :)
        real(8), intent(in) :: fit_matrix(:, :)
        real(8), intent(in) :: basis_mean(:)

        ! Declare outputs
        real(8), intent(out) :: update(size(fit_matrix, 1))
        real(8), intent(out) :: update_mean

        ! Declare local variables
        integer :: prod_idx, covariate, n_samples
        real(8) :: new_node

        n_samples = size(x, 1)
        prod_idx = count(where(:, nbases))
        new_node = nodes(prod_idx, nbases)
        covariate = covariates(prod_idx, nbases)

        update = x(:, covariate) - new_node
        where (x(:, covariate) >= old_node)
            update = old_node - new_node
        elsewhere (x(:, covariate) < new_node)
            update = 0.0d0
        end where

        ! Multiply by parent basis function if not constant
        if (parent_idx /= 0) then
            update = update * (fit_matrix(:, parent_idx) + basis_mean(parent_idx))
        end if

        ! Calculate mean and subtract
        update_mean = sum(update) / n_samples
        update = update - update_mean

    end subroutine update_init

    subroutine calculate_fit_matrix(x, nbases, covariates, nodes, hinges, where, fit_matrix, basis_mean)
        ! Declare inputs
        real(8), intent(in) :: x(:, :)
        integer, intent(in) :: nbases
        integer, intent(in) :: covariates(:, :)
        real(8), intent(in) :: nodes(:, :)
        logical, intent(in) :: hinges(:, :)
        logical, intent(in) :: where(:, :)

        ! Declare outputs
        real(8), intent(out) :: fit_matrix(size(x, 1), nbases)
        real(8), intent(out) :: basis_mean(nbases)

        ! Declare local variables
        integer :: basis_indices(nbases)

        call active_base_indices(where, nbases, basis_indices)
        call data_matrix(x, basis_indices, covariates, nodes, hinges, where, fit_matrix, basis_mean)

    end subroutine calculate_fit_matrix

    subroutine extend_fit_matrix(x, nadditions, fit_matrix, basis_mean, covariates, nodes, hinges, where, &
            fit_matrix_extended, basis_mean_extended)
        ! Declare inputs
        real(8), intent(in) :: x(:, :)
        integer, intent(in) :: nadditions
        real(8), intent(in) :: fit_matrix(:, :)
        real(8), intent(in) :: basis_mean(:)
        integer, intent(in) :: covariates(:, :)
        real(8), intent(in) :: nodes(:, :)
        logical, intent(in) :: hinges(:, :)
        logical, intent(in) :: where(:, :)

        ! Declare outputs
        real(8), intent(out) :: fit_matrix_extended(size(x, 1), size(basis_mean) + nadditions)
        real(8), intent(out) :: basis_mean_extended(size(basis_mean) + nadditions)

        ! Declare local variables
        integer :: indices(size(basis_mean) + nadditions)
        real(8) :: fit_matrix_ext(size(fit_matrix, 1), nadditions)
        real(8) :: basis_mean_ext(nadditions)

        call active_base_indices(where, size(basis_mean), indices)
        if (size(basis_mean) > 0) then
            call data_matrix(x, indices(size(indices) - nadditions:size(indices)), covariates, nodes, hinges, where, &
                    fit_matrix_ext, basis_mean_ext)

            fit_matrix_extended(:, 1:size(fit_matrix, 2)) = fit_matrix
            fit_matrix_extended(:, size(fit_matrix, 2) + 1:size(indices)) = fit_matrix_ext

            basis_mean_extended(1:size(basis_mean)) = basis_mean
            basis_mean_extended(size(basis_mean) + 1:size(indices)) = basis_mean_ext
        else
            call data_matrix(x, indices, covariates, nodes, hinges, where, fit_matrix_ext, basis_mean_ext)
        end if
    end subroutine extend_fit_matrix
end module omars
