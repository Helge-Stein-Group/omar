module backend
    use omp_lib
    implicit none
contains
    subroutine active_base_indices(mask, nbases, result)
        logical, intent(in) :: mask(:, :)
        integer, intent(in) :: nbases

        integer, intent(out) :: result(nbases - 1)

        integer :: j
        logical :: active_base_mask(size(mask, 2))

        active_base_mask = any(mask, dim = 1)

        result = pack([(j, j = 1, size(mask, 2))], active_base_mask)

    end subroutine active_base_indices

    subroutine data_matrix(x, basis_indices, mask, truncated, cov, root, data_matrix_out, data_matrix_mean)
        real(8), intent(in) :: x(:, :)
        integer, intent(in) :: basis_indices(:)
        logical, intent(in) :: mask(:, :)
        logical, intent(in) :: truncated(:, :)
        integer, intent(in) :: cov(:, :)
        real(8), intent(in) :: root(:, :)

        real(8), intent(out) :: data_matrix_out(size(x, 1), size(basis_indices))
        real(8), intent(out) :: data_matrix_mean(size(basis_indices))

        integer :: i, basis_idx, func_idx
        real(8) :: intermediate_result(size(x, 1))

        data_matrix_out = 1.0d0

        do i = 1, size(basis_indices)
            basis_idx = basis_indices(i)
            do func_idx = 1, size(mask, 1)
                if (mask(func_idx, basis_idx)) then
                    intermediate_result = x(:, cov(func_idx, basis_idx)) - root(func_idx, basis_idx)
                    if (truncated(func_idx, basis_idx)) then
                        intermediate_result = max(0.0d0, intermediate_result)
                    end if
                    data_matrix_out(:, i) = data_matrix_out(:, i) * intermediate_result
                end if
            end do
            data_matrix_mean(i) = sum(data_matrix_out(:, i)) / size(x, 1)
            data_matrix_out(:, i) = data_matrix_out(:, i) - data_matrix_mean(i)
        end do
    end subroutine data_matrix

    subroutine covariance_matrix(data_matrix, covariance_matrix_out)
        real(8), intent(in) :: data_matrix(:, :)
        real(8), intent(out) :: covariance_matrix_out(size(data_matrix, 2), size(data_matrix, 2))

        integer :: i

        covariance_matrix_out = matmul(transpose(data_matrix), data_matrix)

        ! Add epsilon to the diagonal
        do i = 1, size(data_matrix, 2)
            covariance_matrix_out(i, i) = covariance_matrix_out(i, i) + 1.0d-8
        end do
    end subroutine covariance_matrix

    subroutine rhs(y, y_mean, data_matrix_in, rhs_out)
        real(8), intent(in) :: y(:)
        real(8), intent(in) :: y_mean
        real(8), intent(in) :: data_matrix_in(:, :)

        real(8), intent(out) :: rhs_out(size(data_matrix_in, 2))
        real(8) :: y_centred(size(y))

        y_centred = y - y_mean
        rhs_out = matmul(transpose(data_matrix_in), y_centred)

    end subroutine rhs

    subroutine coefficients(covariance_matrix, rhs, coefficients_out, chol)
        real(8), intent(in) :: covariance_matrix(:, :)
        real(8), intent(in) :: rhs(:)

        real(8), intent(out) :: coefficients_out(size(rhs))
        real(8), intent(out) :: chol(size(covariance_matrix, 1), size(covariance_matrix, 2))

        integer :: info


        ! Use LAPACK DPOTRF to compute the Cholesky decomposition
        chol = covariance_matrix
        call dpotrf('L', size(chol, 1), chol, size(chol, 1), info)  ! 'L' for lower triangular
        if (info /= 0) then
            print *, "Error during Cholesky decomposition, info = ", info
            stop
        end if

        ! Solve the system using LAPACK's dpotrs
        coefficients_out = rhs
        call dpotrs('L', size(chol, 1), 1, chol, size(chol, 1), coefficients_out, size(chol, 1), info)
        if (info /= 0) then
            print *, "Error during solving linear system with dpotrs, info = ", info
            stop
        end if
    end subroutine coefficients

    subroutine generalised_cross_validation(y, y_mean, data_matrix_in, chol, coefficients_in, penalty, lof)
        real(8), intent(in) :: y(:)
        real(8), intent(in) :: y_mean
        real(8), intent(in) :: data_matrix_in(:, :)
        real(8), intent(in) :: chol(:, :)
        real(8), intent(in) :: coefficients_in(:)
        integer, intent(in) :: penalty

        real(8), intent(out) :: lof

        real(8) :: y_pred(size(y))
        real(8) :: mse
        integer :: rank
        integer :: i
        real(8) :: c_m

        if (size(data_matrix_in, 2) /= 0) then
            y_pred = matmul(data_matrix_in, coefficients_in) + y_mean
            rank = 0
            do i = 1, size(chol, 1)
                if (chol(i, i) /= 0.0d0) then
                    rank = rank + 1
                end if
            end do
            c_m = rank * (1 + penalty) + 1 - penalty

        else
            y_pred = y_mean
            c_m = 1 - penalty
        end if
        mse = sum((y - y_pred) ** 2) / real(size(y, 1))

        if (c_m /= size(y)) then
            lof = mse / (1 - c_m / size(y)) ** 2
        else
            lof = 10d20
        end if

    end subroutine generalised_cross_validation

    subroutine fit(x, y, y_mean, nbases, mask, truncated, cov, root, penalty, &
            data_matrix_out, data_matrix_mean, covariance_matrix_out, rhs_out, chol, coefficients_out, lof)
        real(8), intent(in) :: x(:, :)
        real(8), intent(in) :: y(:)
        real(8), intent(in) :: y_mean
        integer, intent(in) :: nbases
        logical, intent(in) :: mask(:, :)
        logical, intent(in) :: truncated(:, :)
        integer, intent(in) :: cov(:, :)
        real(8), intent(in) :: root(:, :)
        integer, intent(in) :: penalty

        real(8), intent(out) :: data_matrix_out(size(x, 1), nbases - 1)
        real(8), intent(out) :: data_matrix_mean(nbases - 1)
        real(8), intent(out) :: covariance_matrix_out(nbases - 1, nbases - 1)
        real(8), intent(out) :: rhs_out(nbases - 1)
        real(8), intent(out) :: chol(nbases - 1, nbases - 1)
        real(8), intent(out) :: coefficients_out(nbases - 1)
        real(8), intent(out) :: lof

        integer :: indices(nbases - 1)

        call active_base_indices(mask, nbases, indices)
        call data_matrix(x, indices, mask, truncated, cov, root, data_matrix_out, data_matrix_mean)
        call covariance_matrix(data_matrix_out, covariance_matrix_out)
        call rhs(y, y_mean, data_matrix_out, rhs_out)
        call coefficients(covariance_matrix_out, rhs_out, coefficients_out, chol)
        call generalised_cross_validation(y, y_mean, data_matrix_out, chol, coefficients_out, penalty, lof)
    end subroutine fit

    subroutine update_init(x, data_matrix_in, data_matrix_mean, prev_root, parent_idx, nbases, mask, cov, root, &
            update, update_mean)
        real(8), intent(in) :: x(:, :)
        real(8), intent(in) :: data_matrix_in(:, :)
        real(8), intent(in) :: data_matrix_mean(:)
        real(8), intent(in) :: prev_root
        integer, intent(in) :: parent_idx
        integer, intent(in) :: nbases
        logical, intent(in) :: mask(:, :)
        integer, intent(in) :: cov(:, :)
        real(8), intent(in) :: root(:, :)

        real(8), intent(out) :: update(size(data_matrix_in, 1))
        real(8), intent(out) :: update_mean

        integer :: prod_idx
        integer :: new_cov
        real(8) :: new_root

        prod_idx = count(mask(:, nbases)) + 1
        new_root = root(prod_idx, nbases)
        new_cov = cov(prod_idx, nbases)

        update = x(:, new_cov) - new_root
        where (x(:, new_cov) >= prev_root)
            update = prev_root - new_root
        elsewhere (x(:, new_cov) < new_root)
            update = 0.0d0
        end where

        ! Multiply by parent basis function if not constant
        if (parent_idx /= 1) then
            update = update * (data_matrix_in(:, parent_idx - 1) + data_matrix_mean(parent_idx - 1))
        end if

        ! Calculate mean and subtract
        update_mean = sum(update) / size(x, 1)
        update = update - update_mean

    end subroutine update_init

    subroutine update_data_matrix(data_matrix_in, data_matrix_mean, update, update_mean)
        real(8), intent(inout) :: data_matrix_in(:, :)
        real(8), intent(inout) :: data_matrix_mean(:)
        real(8), intent(in) :: update(:)
        real(8), intent(in) :: update_mean

        data_matrix_in(:, size(data_matrix_in, 2)) = data_matrix_in(:, size(data_matrix_in, 2)) + update
        data_matrix_mean(size(data_matrix_in, 2)) = data_matrix_mean(size(data_matrix_in, 2)) + update_mean

    end subroutine update_data_matrix

    subroutine update_covariance_matrix(covariance_matrix_in, data_matrix_in, update, covariance_addition)
        real(8), intent(inout) :: covariance_matrix_in(:, :)
        real(8), intent(in) :: data_matrix_in(:, :)
        real(8), intent(in) :: update(:)

        real(8), intent(out) :: covariance_addition(size(covariance_matrix_in, 2))

        integer :: last

        last = size(covariance_matrix_in, 2)

        covariance_addition(1:last - 1) = matmul(update, data_matrix_in(:, 1:last - 1))
        covariance_addition(last) = 2.0d0 * dot_product(data_matrix_in(:, last), update)
        covariance_addition(last) = covariance_addition(last) - dot_product(update, update)

        covariance_matrix_in(last, 1:last - 1) = covariance_matrix_in(last, 1:last - 1) + covariance_addition(1:last - 1)
        covariance_matrix_in(:, last) = covariance_matrix_in(:, last) + covariance_addition

    end subroutine update_covariance_matrix

    subroutine update_rhs(rhs_in, update, y, y_mean)
        real(8), intent(inout) :: rhs_in(:)
        real(8), intent(in) :: update(:)
        real(8), intent(in) :: y(:)
        real(8), intent(in) :: y_mean

        real(8) :: y_centred(size(y))

        y_centred = y - y_mean
        rhs_in(size(rhs_in)) = rhs_in(size(rhs_in)) + sum(update * y_centred)

    end subroutine update_rhs

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

    subroutine update_cholesky(chol, update_vector, multiplier)
        real(8), intent(inout) :: chol(:, :)
        real(8), intent(in) :: update_vector(:)
        real(8), intent(in) :: multiplier

        real(8) :: diag(size(chol, 1))
        real(8) :: u(size(chol, 1), size(update_vector))
        real(8) :: b(size(chol, 1))
        integer :: i

        do i = 1, size(chol, 1)
            diag(i) = chol(i, i)
        end do

        do i = 1, size(chol, 1)
            chol(:, i) = chol(:, i) / diag(i)
        end do
        diag = diag**2

        u = 0.0
        u(1, :) = update_vector
        u(1, 2:) = u(1, 2:) - update_vector(1) * chol(2:, 1)
        b = 1.0

        do i = 2, size(chol, 1)
            u(i, :) = u(i - 1, :)
            u(i, i + 1:) = u(i, i + 1:) - u(i - 1, i) * chol(i + 1:, i)
            b(i) = b(i - 1) + multiplier * u(i - 1, i - 1)**2 / diag(i - 1)
        end do

        do i = 1, size(chol, 1)
            chol(i, i) = sqrt(diag(i) + multiplier / b(i) * u(i, i)**2)
            chol(i + 1:, i) = chol(i + 1:, i) * chol(i, i)
            chol(i + 1:, i) = chol(i + 1:, i) + multiplier / b(i) * u(i, i) * u(i, i + 1:) / chol(i, i)
        end do
    end subroutine update_cholesky

    subroutine update_coefficients(coefficients_in, chol, covariance_addition, rhs_in)
        real(8), intent(inout) :: coefficients_in(:)
        real(8), intent(inout) :: chol(:, :)
        real(8), intent(in) :: covariance_addition(:)
        real(8), intent(in) :: rhs_in(:)

        real(8) :: eigenvalues(2)
        real(8) :: eigenvectors(2, size(covariance_addition))
        integer :: i
        integer :: info

        call decompose_addition(covariance_addition, eigenvalues, eigenvectors)
        do i = 1, 2
            call update_cholesky(chol, eigenvectors(i, :), eigenvalues(i))
        end do

        coefficients_in = rhs_in
        call dpotrs('L', size(chol, 1), 1, chol, size(chol, 1), coefficients_in, size(chol, 1), info)
        if (info /= 0) then
            print *, "Error during solving linear system with dpotrs, info = ", info
            stop
        end if
    end subroutine update_coefficients

    subroutine update_fit(data_matrix_in, data_matrix_mean, covariance_matrix_in, rhs_in, chol, coefficients_in, &
            x, y, prev_root, parent_idx, y_mean, nbases, penalty, mask, cov, root, lof)
        real(8), intent(inout) :: data_matrix_in(:, :)
        real(8), intent(inout) :: data_matrix_mean(:)
        real(8), intent(inout) :: covariance_matrix_in(:, :)
        real(8), intent(inout) :: rhs_in(:)
        real(8), intent(inout) :: chol(:, :)
        real(8), intent(inout) :: coefficients_in(:)

        real(8), intent(in) :: x(:, :)
        real(8), intent(in) :: y(:)
        real(8), intent(in) :: prev_root
        integer, intent(in) :: parent_idx
        real(8), intent(in) :: y_mean
        integer, intent(in) :: nbases
        integer, intent(in) :: penalty
        logical, intent(in) :: mask(:, :)
        integer, intent(in) :: cov(:, :)
        real(8), intent(in) :: root(:, :)

        real(8), intent(out) :: lof

        real(8) :: update(size(x, 1))
        real(8) :: update_mean
        real(8) :: covariance_addition(size(chol, 1))

        call update_init(x, data_matrix_in, data_matrix_mean, prev_root, parent_idx, nbases, mask, cov, root, &
                update, update_mean)
        call update_data_matrix(data_matrix_in, data_matrix_mean, update, update_mean)
        call update_covariance_matrix(covariance_matrix_in, data_matrix_in, update, covariance_addition)
        call update_rhs(rhs_in, update, y, y_mean)
        call update_coefficients(coefficients_in, chol, covariance_addition, rhs_in)

        call generalised_cross_validation(y, y_mean, data_matrix_in, chol, coefficients_in, penalty, lof)
    end subroutine update_fit

    subroutine argsort(array, indices)
        real(8), intent(in) :: array(:)
        integer, intent(out) :: indices(size(array))
        integer :: i, j, n
        integer :: temp_idx

        n = size(array)
        indices = [(i, i = 1, n)]

        do i = 1, n - 1
            do j = 1, n - i
                if (array(indices(j)) > array(indices(j + 1))) then
                    temp_idx = indices(j)
                    indices(j) = indices(j + 1)
                    indices(j + 1) = temp_idx
                end if
            end do
        end do
    end subroutine argsort

    subroutine add_bases(parent, cov_in, root_in, nbases, mask, truncated, cov, root)
        integer, intent(in) :: parent
        integer, intent(in) :: cov_in
        real(8), intent(in) :: root_in
        integer, intent(in) :: nbases
        logical, intent(inout) :: mask(:, :)
        logical, intent(inout) :: truncated(:, :)
        integer, intent(inout) :: cov(:, :)
        real(8), intent(inout) :: root(:, :)

        integer :: parent_depth

        parent_depth = count(mask(:, parent))

        mask(:, nbases - 1) = mask(:, parent)
        mask(:, nbases) = mask(:, parent)
        mask(parent_depth + 2, nbases - 1:nbases) = .true.

        truncated(:, nbases - 1) = truncated(:, parent)
        truncated(:, nbases) = truncated(:, parent)
        truncated(parent_depth + 2, nbases) = .true.

        cov(:, nbases - 1) = cov(:, parent)
        cov(:, nbases) = cov(:, parent)
        cov(parent_depth + 2, nbases - 1:nbases) = cov_in

        root(:, nbases - 1) = root(:, parent)
        root(:, nbases) = root(:, parent)
        root(parent_depth + 2, nbases) = root_in
    end subroutine add_bases

    subroutine expand_bases(x, y, y_mean, max_nbases, max_ncandidates, aging_factor, penalty, &
            lof, nbases, mask, truncated, cov, root, coefficients_out)
        real(8), intent(in) :: x(:, :)
        real(8), intent(in) :: y(:)
        real(8), intent(in) :: y_mean
        integer, intent(in) :: max_nbases
        integer, intent(in) :: max_ncandidates
        real(8), intent(in) :: aging_factor
        integer, intent(in) :: penalty

        real(8), intent(out) :: lof
        integer, intent(out) :: nbases
        logical, intent(out) :: mask(max_nbases, max_nbases)
        logical, intent(out) :: truncated(max_nbases, max_nbases)
        integer, intent(out) :: cov(max_nbases, max_nbases)
        real(8), intent(out) :: root(max_nbases, max_nbases)
        real(8), intent(out) :: coefficients_out(max_nbases - 1)

        real(8), allocatable :: candidate_queue(:)
        integer :: iteration
        real(8) :: best_lof
        integer :: best_cov
        real(8) :: best_root
        integer :: best_parent
        integer, allocatable :: parents(:)
        integer :: num_pairs
        integer :: pairs(max_ncandidates * size(x, 2), 2)
        integer :: parent
        integer :: cov_idx
        real(8), allocatable :: basis_lofs(:)
        integer :: parent_depth
        integer, allocatable :: indices(:)
        real(8), allocatable :: eligible_roots(:)
        real(8), allocatable :: a_data_matrix_prev(:, :)
        integer :: i, j
        integer :: info
        integer :: root_idx
        real(8), allocatable :: a_data_matrix(:, :)
        real(8), allocatable :: a_data_matrix_mean(:)
        real(8), allocatable :: a_covariance_matrix(:, :)
        real(8), allocatable :: a_rhs(:)
        real(8), allocatable :: a_chol(:, :)
        real(8), allocatable :: a_coefficients(:)
        real(8), allocatable :: candidate_queue_buffer(:)

        nbases = 1
        mask = .false.
        truncated = .false.
        cov = 0
        root = 0d0

        allocate(candidate_queue(1))
        candidate_queue(1) = 0

        do iteration = 1, (max_nbases - 1) / 2
            best_lof = 1d20
            best_cov = -1
            best_root = -1d0
            best_parent = -1

            allocate(parents(size(candidate_queue)))
            call argsort(candidate_queue, parents)

            num_pairs = 0
            do parent = 1, min(max_ncandidates, nbases)
                do cov_idx = 1, size(x, 2)
                    if (all(cov(:, parents(parent)) /= cov_idx .or. .not. mask(:, parents(parent)))) then
                        num_pairs = num_pairs + 1
                        pairs(num_pairs, 1) = parent
                        pairs(num_pairs, 2) = cov_idx
                    end if
                end do
            end do

            allocate(basis_lofs(min(max_ncandidates, nbases)))
            basis_lofs = 1d20

            nbases = nbases + 2
            !$OMP PARALLEL DO DEFAULT(firstprivate) &
            !$OMP& SHARED(parents, pairs, x, y, y_mean, penalty, &
            !$OMP& best_lof, best_cov, best_root, best_parent, basis_lofs)
            do i = 1, num_pairs
                parent = parents(pairs(i, 1))
                cov_idx = pairs(i, 2)

                call add_bases(parent, cov_idx, 0d0, nbases, mask, truncated, cov, root)

                if (parent == 1) then
                    allocate(eligible_roots(size(x, 1)))
                    eligible_roots = x(:, cov_idx)
                else
                    allocate(eligible_roots(count(a_data_matrix_prev(:, parent - 1) > 0)))
                    eligible_roots = x(pack((/(j, j = 1, size(x, 1))/), a_data_matrix_prev(:, parent - 1) > 0), cov_idx)
                end if

                ! Sort the array in descending order using LAPACK's dlasrt
                call dlasrt('D', size(eligible_roots), eligible_roots, info)
                if (info /= 0) then
                    print *, "Sorting failed, info: ", info
                    stop
                end if

                parent_depth = count(mask(:, parent))
                do root_idx = 1, size(eligible_roots)
                    root(parent_depth + 2, nbases) = eligible_roots(root_idx)
                    if (root_idx == 1) then
                        allocate(a_data_matrix(size(x, 1), nbases - 1))
                        allocate(a_data_matrix_mean(nbases - 1))
                        allocate(a_covariance_matrix(nbases - 1, nbases - 1))
                        allocate(a_rhs(nbases - 1))
                        allocate(a_chol(nbases - 1, nbases - 1))
                        allocate(a_coefficients(nbases - 1))
                        call fit(x, y, y_mean, nbases, mask, truncated, cov, root, penalty, &
                                a_data_matrix, a_data_matrix_mean, a_covariance_matrix, a_rhs, a_chol, a_coefficients, &
                                lof)
                    else
                        call update_fit(a_data_matrix, a_data_matrix_mean, a_covariance_matrix, a_rhs, a_chol, &
                                a_coefficients, x, y, eligible_roots(root_idx - 1), parent, &
                                y_mean, nbases, penalty, mask, cov, root, lof)
                    end if
                    !$OMP CRITICAL
                    if (lof < basis_lofs(pairs(i, 1))) then
                        basis_lofs(pairs(i, 1)) = lof
                    end if
                    if (lof < best_lof) then
                        best_lof = lof
                        best_cov = cov_idx
                        best_root = eligible_roots(root_idx)
                        best_parent = parent
                    end if
                    !$OMP END CRITICAL
                end do

                deallocate(eligible_roots)
                deallocate(a_data_matrix)
                deallocate(a_data_matrix_mean)
                deallocate(a_covariance_matrix)
                deallocate(a_rhs)
                deallocate(a_chol)
                deallocate(a_coefficients)
            end do
            !$OMP END PARALLEL DO
            do i = 1, size(candidate_queue)
                if (i <= size(basis_lofs)) then
                    candidate_queue(parents(i)) = basis_lofs(i) - best_lof
                else
                    candidate_queue(parents(i)) = candidate_queue(parents(i)) - aging_factor
                end if
            end do

            if (best_cov /= -1) then
                call add_bases(best_parent, best_cov, best_root, nbases, mask, truncated, cov, root)

                allocate(candidate_queue_buffer(size(candidate_queue)))
                candidate_queue_buffer = candidate_queue
                deallocate(candidate_queue)
                allocate(candidate_queue(nbases))
                candidate_queue(1:size(candidate_queue_buffer)) = candidate_queue_buffer
                candidate_queue(nbases - 1:) = 0

                if (allocated(a_data_matrix_prev)) then
                    deallocate(a_data_matrix_prev)
                end if
                allocate(indices(nbases - 1))
                allocate(a_data_matrix_prev(size(x, 1), nbases - 1))
                allocate(a_data_matrix_mean(nbases - 1))
                call active_base_indices(mask, nbases, indices)
                call data_matrix(x, indices, mask, truncated, cov, root, a_data_matrix_prev, a_data_matrix_mean)
                deallocate(indices)
                deallocate(a_data_matrix_mean)
            else
                print *, "Cannot find additional bases in iteration", iteration, "."
                mask(:, nbases - 1:nbases) = .false.
                nbases = nbases - 2
                exit
            end if
            deallocate(parents)
            deallocate(basis_lofs)
            deallocate(candidate_queue_buffer)
        end do
        deallocate(candidate_queue)

        allocate(a_data_matrix(size(x, 1), nbases - 1))
        allocate(a_data_matrix_mean(nbases - 1))
        allocate(a_covariance_matrix(nbases - 1, nbases - 1))
        allocate(a_rhs(nbases - 1))
        allocate(a_chol(nbases - 1, nbases - 1))
        call fit(x, y, y_mean, nbases, mask, truncated, cov, root, penalty, &
                a_data_matrix, a_data_matrix_mean, a_covariance_matrix, a_rhs, a_chol, &
                coefficients_out, lof)
        deallocate(a_data_matrix)
        deallocate(a_data_matrix_mean)
        deallocate(a_covariance_matrix)
        deallocate(a_rhs)
        deallocate(a_chol)
    end subroutine expand_bases

    subroutine prune_bases(x, y, y_mean, lof, nbases, mask_in, truncated, cov, root, penalty, coefficients_out, &
            mask)
        real(8), intent(in) :: x(:, :)
        real(8), intent(in) :: y(:)
        real(8), intent(in) :: y_mean
        real(8), intent(inout) :: lof
        integer, intent(inout) :: nbases
        logical, intent(in) :: mask_in(:, :)
        logical, intent(in) :: truncated(:, :)
        integer, intent(in) :: cov(:, :)
        real(8), intent(in) :: root(:, :)
        integer, intent(in) :: penalty

        real(8), intent(out) :: coefficients_out(nbases - 1)
        logical, intent(out) :: mask(size(mask_in, 1), size(mask_in, 2)) ! Requires splititng since fortran expects 4 byte for a logical?!

        integer :: best_nbases
        logical :: best_mask(size(mask, 1), size(mask, 2))
        real(8) :: best_lof
        logical :: prev_mask(size(mask, 1), size(mask, 2))
        integer :: iteration
        real(8) :: best_lof_trim
        integer :: removal_idx
        integer, allocatable :: indices(:)
        integer :: i
        real(8), allocatable :: a_data_matrix(:, :)
        real(8), allocatable :: a_data_matrix_mean(:)
        real(8), allocatable :: a_covariance_matrix(:, :)
        real(8), allocatable :: a_rhs(:)
        real(8), allocatable :: a_chol(:, :)
        real(8), allocatable :: a_coefficients(:)

        coefficients_out = 0.0d0
        mask = mask_in

        best_nbases = nbases
        best_mask = mask
        best_lof = lof
        prev_mask = mask

        do iteration = 1, nbases - 1
            best_lof_trim = 1d20
            removal_idx = -1

            allocate(indices(nbases - 1))
            call active_base_indices(mask, nbases, indices)

            allocate(a_data_matrix(size(x, 1), nbases - 2))
            allocate(a_data_matrix_mean(nbases - 2))
            allocate(a_covariance_matrix(nbases - 2, nbases - 2))
            allocate(a_rhs(nbases - 2))
            allocate(a_chol(nbases - 2, nbases - 2))
            allocate(a_coefficients(nbases - 2))
            do i = 1, size(indices)
                mask(:, indices(i)) = .false.
                nbases = nbases - 1

                if (nbases == 1) then
                    call generalised_cross_validation(y, y_mean, a_data_matrix, a_chol, a_coefficients, penalty, lof)
                else
                    call fit(x, y, y_mean, nbases, mask, truncated, cov, root, penalty, &
                            a_data_matrix, a_data_matrix_mean, a_covariance_matrix, a_rhs, a_chol, a_coefficients, &
                            lof)
                end if

                if (lof < best_lof_trim) then
                    best_lof_trim = lof
                    removal_idx = indices(i)
                end if
                if (lof < best_lof) then
                    best_lof = lof
                    best_nbases = nbases
                    best_mask = mask
                end if

                mask(:, indices(i)) = prev_mask(:, indices(i))
                nbases = nbases + 1
            end do
            deallocate(indices)
            deallocate(a_data_matrix)
            deallocate(a_data_matrix_mean)
            deallocate(a_covariance_matrix)
            deallocate(a_rhs)
            deallocate(a_chol)
            deallocate(a_coefficients)

            mask(:, removal_idx) = .false.
            nbases = nbases - 1
            prev_mask = mask
        end do
        mask = best_mask
        nbases = best_nbases

        allocate(a_data_matrix(size(x, 1), nbases - 1))
        allocate(a_data_matrix_mean(nbases - 1))
        allocate(a_covariance_matrix(nbases - 1, nbases - 1))
        allocate(a_rhs(nbases - 1))
        allocate(a_chol(nbases - 1, nbases - 1))
        allocate(a_coefficients(nbases - 1))
        call fit(x, y, y_mean, nbases, mask, truncated, cov, root, penalty, &
                a_data_matrix, a_data_matrix_mean, a_covariance_matrix, a_rhs, a_chol, a_coefficients, &
                lof)
        coefficients_out(:nbases - 1) = a_coefficients
        deallocate(a_data_matrix)
        deallocate(a_data_matrix_mean)
        deallocate(a_covariance_matrix)
        deallocate(a_rhs)
        deallocate(a_chol)
    end subroutine prune_bases

    subroutine find_bases(x, y, y_mean, max_nbases, max_ncandidates, aging_factor, penalty, &
            lof, nbases, mask, truncated, cov, root, coefficients_out)
        real(8), intent(in) :: x(:, :)
        real(8), intent(in) :: y(:)
        real(8), intent(in) :: y_mean
        integer, intent(in) :: max_nbases
        integer, intent(in) :: max_ncandidates
        real(8), intent(in) :: aging_factor
        integer, intent(in) :: penalty

        real(8), intent(out) :: lof
        integer, intent(out) :: nbases
        logical, intent(out) :: mask(max_nbases, max_nbases)
        logical, intent(out) :: truncated(max_nbases, max_nbases)
        integer, intent(out) :: cov(max_nbases, max_nbases)
        real(8), intent(out) :: root(max_nbases, max_nbases)
        real(8), intent(out) :: coefficients_out(max_nbases - 1)

        !logical :: where_temp(max_nbases, max_nbases) ! fuck you fortran
        logical :: mask_in(max_nbases, max_nbases)

        call expand_bases(x, y, y_mean, max_nbases, max_ncandidates, aging_factor, penalty, &
                lof, nbases, mask_in, truncated, cov, root, coefficients_out)
        call prune_bases(x, y, y_mean, lof, nbases, mask_in, truncated, cov, root, penalty, coefficients_out, mask)
    end subroutine find_bases

end module backend
