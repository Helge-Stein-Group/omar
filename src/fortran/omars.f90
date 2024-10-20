module omars
    !$ use omp_lib
    implicit none
    !f2py threadsafef
contains
    subroutine update_cholesky(chol, update_vectors, multipliers)
        real(8), intent(in) :: update_vectors(:, :)
        real(8), intent(in) :: multipliers(2)
        real(8), intent(inout) :: chol(:, :)

        real(8) :: diag(size(chol, 1))
        real(8) :: u(size(chol, 1), size(update_vectors, 2))
        real(8) :: b(size(chol, 1))
        integer :: i, update_idx

        do update_idx = 1, 2
            ! Extract diagonal
            do i = 1, size(chol, 1)
                diag(i) = chol(i, i)
            end do
            ! Divide chol by its diagonal elements (broadcast manually)
            do i = 1, size(chol, 1)
                chol(:, i) = chol(:, i) / diag(i)
            end do
            diag = diag**2

            u = 0.0
            u(1, :) = update_vectors(update_idx, :)
            u(1, 2:) = u(1, 2:) - update_vectors(update_idx, 1) * chol(2:, 1)
            b = 1.0

            do i = 2, size(chol, 1)
                u(i, :) = u(i - 1, :)
                u(i, i + 1:) = u(i, i + 1:) - u(i - 1, i) * chol(i + 1:, i)
                b(i) = b(i - 1) + multipliers(update_idx) * u(i - 1, i - 1)**2 / diag(i - 1)
            end do

            do i = 1, size(chol, 1)
                chol(i, i) = sqrt(diag(i) + multipliers(update_idx) / b(i) * u(i, i)**2)
                chol(i + 1:, i) = chol(i + 1:, i) * chol(i, i)
                chol(i + 1:, i) = chol(i + 1:, i) + multipliers(update_idx) / b(i) * u(i, i) * u(i, i + 1:) / chol(i, i)
            end do
        end do
    end subroutine update_cholesky

    subroutine decompose_addition(covariance_addition, eigenvalues, eigenvectors)
        real(8), intent(in) :: covariance_addition(:)

        real(8), intent(out) :: eigenvalues(2)
        real(8), intent(out) :: eigenvectors(2, size(covariance_addition))

        real(8) :: eigenvalue_intermediate
        integer :: n, i

        n = size(covariance_addition)

        eigenvalue_intermediate = sqrt(covariance_addition(n)**2 + 4 * sum(covariance_addition(1:n - 1)**2))

        eigenvalues(1) = (covariance_addition(n) + eigenvalue_intermediate) / 2
        eigenvalues(2) = (covariance_addition(n) - eigenvalue_intermediate) / 2

        do i = 1, 2
            eigenvectors(i, 1:n - 1) = covariance_addition(1:n - 1) / eigenvalues(i)
            eigenvectors(i, n) = 1.0d0
            eigenvectors(i, :) = eigenvectors(i, :) / sqrt(sum(eigenvectors(i, :)**2))
        end do

    end subroutine decompose_addition

    subroutine active_base_indices(where, nbases, result)
        integer, intent(in) :: nbases
        logical, intent(in) :: where(:, :)

        integer, intent(out) :: result(nbases - 1)

        integer :: j, n_cols
        logical :: col_has_true(size(where, 2))

        n_cols = size(where, 2)

        col_has_true = any(where, dim = 1)

        result = pack([(j, j = 1, n_cols)], col_has_true)

    end subroutine active_base_indices

    subroutine data_matrix(x, basis_indices, covariates, nodes, hinges, where, result, result_mean)
        real(8), intent(in) :: x(:, :)
        integer, intent(in) :: basis_indices(:)
        integer, intent(in) :: covariates(:, :)
        real(8), intent(in) :: nodes(:, :)
        logical, intent(in) :: hinges(:, :)
        logical, intent(in) :: where(:, :)

        real(8), intent(out) :: result(size(x, 1), size(basis_indices))
        real(8), intent(out) :: result_mean(size(basis_indices))

        integer :: i, basis_idx, func_idx
        real(8) :: intermediate_result(size(x, 1))

        ! Initialize result
        result = 1.0d0

        ! Perform the calculation
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
    end subroutine data_matrix

    subroutine update_init(x, old_node, parent_idx, nbases, covariates, nodes, where, fit_matrix, basis_mean, update, &
            update_mean)
        real(8), intent(in) :: x(:, :)
        real(8), intent(in) :: old_node
        integer, intent(in) :: parent_idx
        integer, intent(in) :: nbases
        integer, intent(in) :: covariates(:, :)
        real(8), intent(in) :: nodes(:, :)
        logical, intent(in) :: where(:, :)
        real(8), intent(in) :: fit_matrix(:, :)
        real(8), intent(in) :: basis_mean(:)

        real(8), intent(out) :: update(size(fit_matrix, 1))
        real(8), intent(out) :: update_mean

        integer :: prod_idx, covariate, n_samples
        real(8) :: new_node

        n_samples = size(x, 1)
        prod_idx = count(where(:, nbases)) + 1
        new_node = nodes(prod_idx, nbases)
        covariate = covariates(prod_idx, nbases) + 1

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
        real(8), intent(in) :: x(:, :)
        integer, intent(in) :: nbases
        integer, intent(in) :: covariates(:, :)
        real(8), intent(in) :: nodes(:, :)
        logical, intent(in) :: hinges(:, :)
        logical, intent(in) :: where(:, :)

        real(8), intent(out) :: fit_matrix(size(x, 1), nbases - 1)
        real(8), intent(out) :: basis_mean(nbases - 1)

        integer :: basis_indices(nbases - 1)

        call active_base_indices(where, nbases, basis_indices)
        call data_matrix(x, basis_indices, covariates, nodes, hinges, where, fit_matrix, basis_mean)

    end subroutine calculate_fit_matrix

    subroutine extend_fit_matrix(x, nadditions, fit_matrix, basis_mean, covariates, nodes, hinges, where, &
            fit_matrix_extended, basis_mean_extended)
        real(8), intent(in) :: x(:, :)
        integer, intent(in) :: nadditions
        real(8), intent(in) :: fit_matrix(:, :)
        real(8), intent(in) :: basis_mean(:)
        integer, intent(in) :: covariates(:, :)
        real(8), intent(in) :: nodes(:, :)
        logical, intent(in) :: hinges(:, :)
        logical, intent(in) :: where(:, :)

        real(8), intent(out) :: fit_matrix_extended(size(x, 1), size(basis_mean) + nadditions)
        real(8), intent(out) :: basis_mean_extended(size(basis_mean) + nadditions)

        integer :: indices(size(basis_mean) + nadditions)
        real(8) :: fit_matrix_ext(size(fit_matrix, 1), nadditions)
        real(8) :: basis_mean_ext(nadditions)

        call active_base_indices(where, size(basis_mean), indices)
        call data_matrix(x, indices(size(indices) - nadditions + 1:size(indices)), covariates, nodes, hinges, where, &
                fit_matrix_ext, basis_mean_ext)

        fit_matrix_extended(:, 1:size(fit_matrix, 2)) = fit_matrix
        fit_matrix_extended(:, size(fit_matrix, 2) + 1:size(fit_matrix_extended, 2)) = fit_matrix_ext

        basis_mean_extended(1:size(basis_mean)) = basis_mean
        basis_mean_extended(size(basis_mean) + 1:size(indices)) = basis_mean_ext
    end subroutine extend_fit_matrix

    subroutine update_fit_matrix(fit_matrix, basis_mean, update, update_mean)
        real(8), intent(in) :: update(:)
        real(8), intent(in) :: update_mean

        real(8), intent(inout) :: fit_matrix(:, :)
        real(8), intent(inout) :: basis_mean(:)

        fit_matrix(:, size(fit_matrix, 2)) = fit_matrix(:, size(fit_matrix, 2)) + update
        basis_mean(size(fit_matrix, 2)) = basis_mean(size(fit_matrix, 2)) + update_mean

    end subroutine update_fit_matrix

    subroutine calculate_covariance_matrix(fit_matrix, covariance_matrix)
        real(8), intent(in) :: fit_matrix(:, :)
        real(8), intent(out) :: covariance_matrix(size(fit_matrix, 2), size(fit_matrix, 2))

        integer :: i

        covariance_matrix = matmul(transpose(fit_matrix), fit_matrix)

        ! Add epsilon to the diagonal
        do i = 1, size(fit_matrix, 2)
            covariance_matrix(i, i) = covariance_matrix(i, i) + 1.0d-8
        end do
    end subroutine calculate_covariance_matrix

    subroutine extend_covariance_matrix(covariance_matrix, nadditions, fit_matrix, covariance_matrix_extended)
        real(8), intent(in) :: covariance_matrix(:, :)
        integer, intent(in) :: nadditions
        real(8), intent(in) :: fit_matrix(:, :)

        real(8), intent(out) :: covariance_matrix_extended(size(covariance_matrix, 1) + nadditions, &
                size(covariance_matrix, 1) + nadditions)

        real(8) :: covariance_extension(size(covariance_matrix, 1) + nadditions, nadditions)
        integer :: i, n_cov, n_ext

        n_cov = size(covariance_matrix, 1)
        n_ext = size(fit_matrix, 2)
        covariance_extension = matmul(transpose(fit_matrix), fit_matrix(:, n_cov + 1:n_ext))

        covariance_matrix_extended = 0.0d0
        covariance_matrix_extended(1:n_cov, 1:n_cov) = covariance_matrix
        covariance_matrix_extended(1:n_cov, n_cov + 1:n_ext) = covariance_extension(1:n_cov, :)
        covariance_matrix_extended(n_cov + 1:n_ext, :) = transpose(covariance_extension)

        do i = 0, nadditions - 1
            covariance_matrix_extended(n_ext - i, n_ext - i) = covariance_matrix_extended(n_ext - i, n_ext - i) + 1.0d-8
        end do

    end subroutine extend_covariance_matrix

    subroutine update_covariance_matrix(covariance_matrix, update, fit_matrix, covariance_addition)
        real(8), intent(in) :: update(:)
        real(8), intent(in) :: fit_matrix(:, :)
        real(8), intent(inout) :: covariance_matrix(:, :)

        real(8), intent(out) :: covariance_addition(size(covariance_matrix, 2))

        integer :: last

        last = size(covariance_matrix, 2)

        covariance_addition(1:last - 1) = matmul(update, fit_matrix(:, 1:last - 1))
        covariance_addition(last) = 2.0d0 * dot_product(fit_matrix(:, last), update)
        covariance_addition(last) = covariance_addition(last) - dot_product(update, update)

        covariance_matrix(last, 1:last - 1) = covariance_matrix(last, 1:last - 1) + covariance_addition(1:last - 1)
        covariance_matrix(:, last) = covariance_matrix(:, last) + covariance_addition

    end subroutine update_covariance_matrix

    subroutine calculate_right_hand_side(y, y_mean, fit_matrix, right_hand_side)
        real(8), intent(in) :: y(:)
        real(8), intent(in) :: y_mean
        real(8), intent(in) :: fit_matrix(:, :)

        real(8), intent(out) :: right_hand_side(size(fit_matrix, 2))
        real(8) :: y_centred(size(y))

        y_centred = y - y_mean
        right_hand_side = matmul(transpose(fit_matrix), y_centred)

    end subroutine calculate_right_hand_side

    subroutine extend_right_hand_side(right_hand_side, y, y_mean, fit_matrix, nadditions, extended_right_hand_side)
        real(8), intent(in) :: right_hand_side(:)
        real(8), intent(in) :: y(:)
        real(8), intent(in) :: y_mean
        real(8), intent(in) :: fit_matrix(:, :)
        integer, intent(in) :: nadditions
        real(8), intent(out) :: extended_right_hand_side(size(right_hand_side) + nadditions)

        real(8) :: y_centred(size(y))

        y_centred = y - y_mean
        extended_right_hand_side(1:size(right_hand_side)) = right_hand_side
        extended_right_hand_side(size(right_hand_side) + 1:size(extended_right_hand_side)) = matmul(transpose(&
                fit_matrix(:, size(extended_right_hand_side) - nadditions + 1:size(extended_right_hand_side))), y_centred)

    end subroutine extend_right_hand_side
    subroutine update_right_hand_side(right_hand_side, y, y_mean, update)
        real(8), intent(inout) :: right_hand_side(:)
        real(8), intent(in) :: y(:)
        real(8), intent(in) :: y_mean
        real(8), intent(in) :: update(:)

        real(8) :: y_centred(size(y))

        y_centred = y - y_mean
        right_hand_side(size(right_hand_side)) = right_hand_side(size(right_hand_side)) + sum(update * y_centred)

    end subroutine update_right_hand_side
    subroutine generalised_cross_validation(y, y_mean, fit_matrix, coefficients, nbases, smoothness, lof)
        real(8), intent(in) :: y(:)
        real(8), intent(in) :: y_mean
        real(8), intent(in) :: fit_matrix(:, :)
        real(8), intent(in) :: coefficients(:)
        integer, intent(in) :: nbases
        integer, intent(in) :: smoothness

        real(8), intent(out) :: lof

        real(8) :: y_pred(size(y))
        real(8) :: mse
        real(8) :: c_m

        y_pred = matmul(fit_matrix, coefficients) + y_mean
        mse = sum((y - y_pred) ** 2)

        c_m = nbases + 1 + smoothness * (nbases - 1)
        lof = mse / real(size(y, 1)) / (1.0d0 - c_m / real(size(y, 1)) + 1.0d-6) ** 2

    end subroutine generalised_cross_validation
    subroutine fit(x, y, nbases, covariates, nodes, hinges, where, smoothness, &
            lof, coefficients, fit_matrix, basis_mean, covariance_matrix, chol, right_hand_side, y_mean)
        real(8), intent(in) :: x(:, :)               ! Data points
        real(8), intent(in) :: y(:)                  ! Target values
        integer, intent(in) :: nbases                ! Number of basis functions
        integer, intent(in) :: covariates(:, :)      ! Covariates of the basis
        real(8), intent(in) :: nodes(:, :)           ! Nodes of the basis
        logical, intent(in) :: hinges(:, :)          ! Hinges of the basis
        logical, intent(in) :: where(:, :)           ! Signals product length of basis
        integer, intent(in) :: smoothness            ! Cost for each basis optimization

        real(8), intent(out) :: lof                  ! Generalized cross-validation criterion
        real(8), intent(out) :: coefficients(nbases - 1)      ! Coefficients of the basis
        real(8), intent(out) :: fit_matrix(size(x, 1), nbases - 1)     ! Fit matrix
        real(8), intent(out) :: basis_mean(nbases - 1)        ! Basis mean
        real(8), intent(out) :: covariance_matrix(nbases - 1, nbases - 1) ! Covariance matrix
        real(8), intent(out) :: chol(nbases - 1, nbases - 1)           ! Cholesky decomposition of the covariance matrix
        real(8), intent(out) :: right_hand_side(nbases - 1)   ! Right hand side of the least-squares problem
        real(8), intent(out) :: y_mean               ! Mean of the target values

        integer :: info

        if (nbases > 1) then
            call calculate_fit_matrix(x, nbases, covariates, nodes, hinges, where, fit_matrix, basis_mean)
            call calculate_covariance_matrix(fit_matrix, covariance_matrix)
            call calculate_right_hand_side(y, y_mean, fit_matrix, right_hand_side)

            ! Use LAPACK DPOTRF to compute the Cholesky decomposition
            chol = covariance_matrix
            call dpotrf('L', size(chol, 1), chol, size(chol, 1), info)  ! 'L' for lower triangular

            if (info /= 0) then
                print *, "Error during Cholesky decomposition, info = ", info
                stop
            end if

            ! Solve the system using LAPACK's dpotrs
            coefficients = right_hand_side
            call dpotrs('L', size(chol, 1), 1, chol, size(chol, 1), coefficients, size(chol, 1), info)

            if (info /= 0) then
                print *, "Error during solving linear system with dpotrs, info = ", info
                stop
            end if

            call generalised_cross_validation(y, y_mean, fit_matrix, coefficients, nbases, smoothness, lof)
        else
            lof = sum(y) / real(size(y))   ! Mean of y
            coefficients = 0.0d0
            fit_matrix = 0.0d0
            basis_mean = 0.0d0
            covariance_matrix = 0.0d0
            chol = 0.0d0
            right_hand_side = 0.0d0
            y_mean = lof
        end if

    end subroutine fit

    subroutine extend_fit(x, y, nbases, covariates, nodes, hinges, where, &
            smoothness, nadditions, fit_matrix, basis_mean, &
            covariance_matrix, right_hand_side, y_mean, &
            lof, coefficients, fit_matrix_ext, basis_mean_ext, &
            covariance_matrix_ext, chol, right_hand_side_ext)
        real(8), intent(in) :: x(:, :)
        real(8), intent(in) :: y(:)
        integer, intent(in) :: nbases
        integer, intent(in) :: covariates(:, :)
        real(8), intent(in) :: nodes(:, :)
        logical, intent(in) :: hinges(:, :)
        logical, intent(in) :: where(:, :)
        integer, intent(in) :: nadditions
        integer, intent(in) :: smoothness
        real(8), intent(in) :: fit_matrix(:, :)
        real(8), intent(in) :: basis_mean(:)
        real(8), intent(in) :: covariance_matrix(:, :)
        real(8), intent(in) :: right_hand_side(:)
        real(8), intent(in) :: y_mean


        real(8), intent(out) :: lof
        real(8), intent(out) :: coefficients(size(basis_mean) + nadditions)
        real(8), intent(out) :: fit_matrix_ext(size(x, 1), size(fit_matrix, 2) + nadditions)
        real(8), intent(out) :: basis_mean_ext(size(basis_mean) + nadditions)
        real(8), intent(out) :: covariance_matrix_ext(size(covariance_matrix, 1) + nadditions, size(covariance_matrix, 2)&
                + nadditions)
        real(8), intent(out) :: chol(size(covariance_matrix, 1) + nadditions, size(covariance_matrix, 2) + nadditions)
        real(8), intent(out) :: right_hand_side_ext(size(right_hand_side) + nadditions)

        integer :: info

        call extend_fit_matrix(x, nadditions, fit_matrix, basis_mean, covariates, nodes, hinges, where, &
                fit_matrix_ext, basis_mean_ext)

        call extend_covariance_matrix(covariance_matrix, nadditions, fit_matrix_ext, covariance_matrix_ext)

        call extend_right_hand_side(right_hand_side, y, y_mean, fit_matrix_ext, nadditions, right_hand_side_ext)

        ! Cholesky decomposition using LAPACK
        chol = covariance_matrix_ext
        call dpotrf('L', size(chol, 1), chol, size(chol, 1), info)  ! 'L' for lower triangular

        if (info /= 0) then
            print *, "Error during Cholesky decomposition, info = ", info
            stop
        end if

        coefficients = right_hand_side_ext
        call dpotrs('L', size(chol, 1), 1, chol, size(chol, 1), coefficients, size(chol, 1), info)
        if (info /= 0) then
            print *, "Error during solving linear system with dpotrs, info = ", info
            stop
        end if
        call generalised_cross_validation(y, y_mean, fit_matrix_ext, coefficients, nbases, smoothness, lof)

    end subroutine extend_fit
    subroutine update_fit(x, y, nbases, covariates, nodes, where, smoothness, &
            fit_matrix, basis_mean, covariance_matrix, right_hand_side, &
            y_mean, old_node, parent_idx, chol, lof, coefficients)
        real(8), intent(in) :: x(:, :)
        real(8), intent(in) :: y(:)
        integer, intent(in) :: nbases
        integer, intent(in) :: covariates(:, :)
        real(8), intent(in) :: nodes(:, :)
        logical, intent(in) :: where(:, :)
        integer, intent(in) :: smoothness
        real(8), intent(in) :: old_node
        integer, intent(in) :: parent_idx

        real(8), intent(inout) :: fit_matrix(:, :)
        real(8), intent(inout) :: basis_mean(:)
        real(8), intent(inout) :: covariance_matrix(:, :)
        real(8), intent(inout) :: right_hand_side(:)
        real(8), intent(inout) :: y_mean
        real(8), intent(inout) :: chol(:, :)

        real(8), intent(out) :: lof
        real(8), intent(out) :: coefficients(nbases - 1)

        integer :: info
        real(8) :: update(size(x, 1))
        real(8) :: update_mean
        real(8) :: covariance_addition(size(chol, 1))
        real(8) :: eigenvalues(2)
        real(8) :: eigenvectors(size(chol, 1), 2)

        call update_init(x, old_node, parent_idx, nbases, covariates, nodes, where, &
                fit_matrix, basis_mean, update, update_mean)
        call update_fit_matrix(fit_matrix, basis_mean, update, update_mean)
        call update_covariance_matrix(covariance_matrix, update, fit_matrix, covariance_addition)

        if (any(covariance_addition /= 0.0d0)) then
            call decompose_addition(covariance_addition, eigenvalues, eigenvectors)
            call update_cholesky(chol, eigenvectors, eigenvalues)
        end if

        call update_right_hand_side(right_hand_side, y, y_mean, update)

        coefficients = right_hand_side
        call dpotrs('L', size(chol, 1), 1, chol, size(chol, 1), coefficients, size(chol, 1), info)
        if (info /= 0) then
            print *, "Error during solving linear system with dpotrs, info = ", info
            stop
        end if

        call generalised_cross_validation(y, y_mean, fit_matrix, coefficients, nbases, smoothness, lof)
    end subroutine update_fit
    subroutine shrink_fit(y, y_mean, nbases, smoothness, removal_idx, &
            fit_matrix, basis_mean, covariance_matrix, right_hand_side, &
            lof, coefficients, fit_matrix_out, basis_mean_out, &
            covariance_matrix_out, chol_out, right_hand_side_out)

        real(8), intent(in) :: y(:)
        real(8), intent(in) :: y_mean
        integer, intent(in) :: nbases
        integer, intent(in) :: smoothness
        integer, intent(in) :: removal_idx
        real(8), intent(inout) :: fit_matrix(:, :)
        real(8), intent(inout) :: basis_mean(:)
        real(8), intent(inout) :: covariance_matrix(:, :)
        real(8), intent(inout) :: right_hand_side(:)

        real(8), intent(out) :: lof
        real(8), intent(out) :: coefficients(size(basis_mean) - 1)
        real(8), intent(out) :: fit_matrix_out(size(fit_matrix, 1), size(fit_matrix, 2) - 1)
        real(8), intent(out) :: basis_mean_out(size(basis_mean) - 1)
        real(8), intent(out) :: covariance_matrix_out(size(covariance_matrix, 1) - 1, size(covariance_matrix, 2) - 1)
        real(8), intent(out) :: chol_out(size(covariance_matrix, 1) - 1, size(covariance_matrix, 2) - 1)
        real(8), intent(out) :: right_hand_side_out(size(right_hand_side, 1) - 1)

        integer :: info
        if (removal_idx /= 0) then
            fit_matrix_out(:, 1:removal_idx) = fit_matrix(:, 1:removal_idx)
            basis_mean_out(1:removal_idx) = basis_mean(1:removal_idx)
            covariance_matrix_out(1:removal_idx, 1:removal_idx) = covariance_matrix(1:removal_idx, 1:removal_idx)

            right_hand_side_out(1:removal_idx) = right_hand_side(1:removal_idx)
        end if
        if (removal_idx /= size(fit_matrix, 2) - 1) then
            fit_matrix_out(:, removal_idx + 1:) = fit_matrix(:, removal_idx + 2:)
            basis_mean_out(removal_idx + 1:) = basis_mean(removal_idx + 2:)
            covariance_matrix_out(removal_idx + 1:, removal_idx + 1:) = covariance_matrix(removal_idx + 2:, removal_idx + 2:)
            right_hand_side_out(removal_idx + 1:) = right_hand_side(removal_idx + 2:)
        end if

        if (removal_idx /= 0 .and. removal_idx /= size(fit_matrix, 2) - 1) then
            covariance_matrix_out(1:removal_idx, removal_idx + 1:) = covariance_matrix(1:removal_idx, removal_idx + 2:)
            covariance_matrix_out(removal_idx + 1:, 1:removal_idx) = covariance_matrix(removal_idx + 2:, 1:removal_idx)
        end if

        chol_out = covariance_matrix_out
        call dpotrf('L', size(chol_out, 1), chol_out, size(chol_out, 1), info)
        if (info /= 0) then
            print *, "Cholesky decomposition failed, info: ", info
            stop
        end if

        coefficients = right_hand_side_out
        call dpotrs('L', size(chol_out, 1), 1, chol_out, size(chol_out, 1), coefficients, size(chol_out, 1), info)
        if (info /= 0) then
            print *, "Error during solving linear system with dpotrs, info = ", info
            stop
        end if

        ! Calculate the generalised cross-validation criterion
        call generalised_cross_validation(y, y_mean, fit_matrix_out, coefficients, nbases, smoothness, lof)
    end subroutine shrink_fit
    subroutine expand_bases(x, y, max_nbases, smoothness, max_ncandidates, aging_factor, nbases, covariates, nodes, &
            hinges, where, lof, coefficients, fit_matrix, basis_mean, covariance_matrix, right_hand_side, y_mean)
        real(8), intent(in) :: x(:, :)
        real(8), intent(in) :: y(:)
        integer, intent(in) :: max_nbases
        integer, intent(in) :: smoothness
        integer, intent(in) :: max_ncandidates
        real(8), intent(in) :: aging_factor

        integer, intent(out) :: nbases
        integer, intent(out) :: covariates(max_nbases, max_nbases)
        real(8), intent(out) :: nodes(max_nbases, max_nbases)
        logical, intent(out) :: hinges(max_nbases, max_nbases)
        logical, intent(out) :: where(max_nbases, max_nbases)
        real(8), intent(out) :: lof
        real(8), intent(out) :: coefficients(max_nbases - 1)
        real(8), intent(out) :: fit_matrix(size(x, 1), max_nbases - 1)
        real(8), intent(out) :: basis_mean(max_nbases - 1)
        real(8), intent(out) :: covariance_matrix(max_nbases - 1, max_nbases - 1)
        real(8), intent(out) :: right_hand_side(max_nbases - 1)
        real(8), intent(out) :: y_mean

        integer :: candidate_queue(max_ncandidates)
        integer :: iteration
        integer :: i
        real(8) :: best_lof
        integer :: best_covariate
        real(8) :: best_node
        integer :: candidate_idx
        integer :: parent_idx
        real(8) :: basis_lof
        integer :: parent_depth
        integer :: cov
        real(8), allocatable :: eligible_knots(:)
        integer :: info
        real(8), allocatable :: coefficients_a(:)
        real(8), allocatable :: fit_matrix_a(:, :)
        real(8), allocatable :: basis_mean_a(:)
        real(8), allocatable :: covariance_matrix_a(:, :)
        real(8), allocatable :: chol_a(:, :)
        real(8), allocatable :: right_hand_side_a(:)

        nbases = 1
        covariates = 0d0
        nodes = 0d0
        hinges = .false.
        where = .false.

        lof = 0
        y_mean = sum(y) / real(size(y, 1))
        candidate_queue = -1d0
        candidate_queue(1) = 1

        do iteration = 1, max_nbases / 2
            do i = 1, count(candidate_queue /= -1)
                parent_idx = candidate_queue(i)
                basis_lof = 1d10
                parent_depth = count(where(:, parent_idx))
                do cov = 1, size(x, 2)
                    if (all(covariates(:, parent_idx) /= cov)) then
                        if (parent_idx == 0) then
                            allocate(eligible_knots(size(x, 1)))
                            eligible_knots = x(:, cov)
                        else
                            allocate(eligible_knots(count(fit_matrix(:, parent_idx - 1) > 0)))
                            eligible_knots = x(pack((/(i, i = 1, size(fit_matrix, 1))/), fit_matrix(:, parent_idx - 1) > 0), cov)
                        end if
                        ! Sort the array in descending order using LAPACK's dlasrt
                        call dlasrt('D', size(eligible_knots), eligible_knots, info)
                        if (info /= 0) then
                            print *, "Sorting failed, info: ", info
                            stop
                        end if

                        covariates(:, nbases + 1) = covariates(:, parent_idx)
                        covariates(:, nbases + 2) = covariates(:, parent_idx)
                        covariates(parent_depth + 1, nbases + 1:nbases + 2) = cov
                        nodes(:, nbases + 1) = nodes(:, parent_idx)
                        nodes(:, nbases + 2) = nodes(:, parent_idx)
                        nodes(parent_depth + 1, nbases + 2) = eligible_knots(1)
                        hinges(:, nbases + 1) = hinges(:, parent_idx)
                        hinges(:, nbases + 2) = hinges(:, parent_idx)
                        hinges(parent_depth + 1, nbases + 2) = .true.
                        where(:, nbases + 1) = where(:, parent_idx)
                        where(:, nbases + 2) = where(:, parent_idx)
                        where(parent_depth + 1, nbases + 1:nbases + 2) = .true.
                        nbases = nbases + 2

                        if (nbases == 3) then
                            allocate(coefficients_a(nbases - 1))
                            allocate(fit_matrix_a(size(x, 1), nbases - 1))
                            allocate(basis_mean_a(nbases - 1))
                            allocate(covariance_matrix_a(nbases - 1, nbases - 1))
                            allocate(chol_a(nbases - 1, nbases - 1))
                            allocate(right_hand_side_a(nbases - 1))
                        end if
                    end if
                end do
            end do
        end do

    end subroutine expand_bases
end module omars
