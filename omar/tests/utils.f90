module utils
    implicit none
    private

    integer, parameter :: N_SAMPLES = 100
    integer, parameter :: DIM = 2
    real(8), parameter :: TOLERANCE = 1d-8

    public :: N_SAMPLES, DIM, TOLERANCE
    public :: generate_data, reference_model
    public :: reference_data_matrix, reference_covariance_matrix, reference_rhs
    public :: compare_real_arrays, compare_real_scalars, compare_int_arrays, &
               compare_real_matrices
    public :: print_model_formula

contains

    subroutine generate_data(x, y, y_true)
        real(8), intent(out) :: x(N_SAMPLES, DIM)
        real(8), intent(out) :: y(N_SAMPLES)
        real(8), intent(out) :: y_true(N_SAMPLES)
        real(8) :: noise(N_SAMPLES)

        call random_seed()
        call random_number(x)
        ! Scale to normal(2, 1) approx
        x = (x - 0.5) * sqrt(12.0) * 1.0 + 2.0

        y_true = x(:, 1) + &
                 max(0.0d0, x(:, 1) - 1.0d0) + &
                 max(0.0d0, x(:, 1) - 1.0d0) * x(:, 2) + &
                 max(0.0d0, x(:, 1) - 1.0d0) * max(0.0d0, x(:, 2) - 0.8d0)

        call random_number(noise)
        ! Scale to normal(0, 0.12) approx
        noise = (noise - 0.5) * sqrt(12.0) * 0.12

        y = y_true + noise
    end subroutine generate_data

    subroutine reference_model(x, nbases, mask, truncated, cov, root)
        real(8), intent(in) :: x(N_SAMPLES, DIM)
        integer, intent(out) :: nbases
        logical, intent(out) :: mask(5, 5)
        logical, intent(out) :: truncated(5, 5)
        integer, intent(out) :: cov(5, 5)
        real(8), intent(out) :: root(5, 5)

        real(8) :: x1_val, x08_val
        integer :: x1_idx(1), x08_idx(1)

        x1_idx = minloc(abs(x(:, 1) - 1.0d0), dim=1)
        x1_val = x(x1_idx(1), 1)
        x08_idx = minloc(abs(x(:, 2) - 0.8d0), dim=1)
        x08_val = x(x08_idx(1), 2)

        nbases = 5
        mask = .false.
        truncated = .false.
        cov = 0
        root = 0.0d0

        mask(2, 2:3) = .true.
        mask(2:3, 4:5) = .true.
        truncated(2, 3:5) = .true.
        truncated(3, 5) = .true. 
        cov(2, 2:5) = 1
        cov(3, 4:5) = 2
        root(2, 3:5) = x1_val
        root(3, 5) = x08_val

    end subroutine reference_model

    subroutine reference_data_matrix(x, ref_data_matrix, ref_data_matrix_mean)
        real(8), intent(in) :: x(N_SAMPLES, DIM)
        real(8), intent(out) :: ref_data_matrix(N_SAMPLES, 4)
        real(8), intent(out) :: ref_data_matrix_mean(4)

        real(8) :: x1_val, x08_val
        integer :: x1_idx(1), x08_idx(1)
        integer :: i

        x1_idx = minloc(abs(x(:, 1) - 1.0d0), dim=1)
        x1_val = x(x1_idx(1), 1)
        x08_idx = minloc(abs(x(:, 2) - 0.8d0), dim=1)
        x08_val = x(x08_idx(1), 2)

        ref_data_matrix(:, 1) = x(:, 1)
        ref_data_matrix(:, 2) = max(0.0d0, x(:, 1) - x1_val)
        ref_data_matrix(:, 3) = max(0.0d0, x(:, 1) - x1_val) * x(:, 2)
        ref_data_matrix(:, 4) = max(0.0d0, x(:, 1) - x1_val) * max(0.0d0, x(:, 2) - x08_val)

        do i = 1, 4
            ref_data_matrix_mean(i) = sum(ref_data_matrix(:, i)) / size(x, 1)
            ref_data_matrix(:, i) = ref_data_matrix(:, i) - ref_data_matrix_mean(i)
        end do

    end subroutine reference_data_matrix

    subroutine reference_covariance_matrix(ref_data_matrix, ref_cov_matrix)
        real(8), intent(in) :: ref_data_matrix(N_SAMPLES, 4)
        real(8), intent(out) :: ref_cov_matrix(4, 4)
        integer :: i

        ref_cov_matrix = matmul(transpose(ref_data_matrix), ref_data_matrix)
        do i = 1, 4
            ref_cov_matrix(i, i) = ref_cov_matrix(i, i) + 1.0d-8
        end do
    end subroutine reference_covariance_matrix

    subroutine reference_rhs(y, ref_data_matrix, ref_rhs_out)
        real(8), intent(in) :: y(N_SAMPLES)
        real(8), intent(in) :: ref_data_matrix(N_SAMPLES, 4)
        real(8), intent(out) :: ref_rhs_out(4)
        real(8) :: y_mean

        y_mean = sum(y) / N_SAMPLES
        ref_rhs_out = matmul(transpose(ref_data_matrix), y - y_mean)
    end subroutine reference_rhs

    subroutine compare_real_arrays(arr1, arr2, test_name)
        real(8), intent(in) :: arr1(:), arr2(:)
        character(len=*), intent(in) :: test_name
        if (size(arr1) /= size(arr2)) then
            print *, test_name, ": FAILED (different sizes)"
            return
        end if
        if (all(abs(arr1 - arr2) < TOLERANCE)) then
            print *, test_name, ": PASSED"
        else
            print *, test_name, ": FAILED"
            print '(A, *(F8.2,1X))', "Array 1:", arr1
            print '(A, *(F8.2,1X))', "Array 2:", arr2
        end if
    end subroutine compare_real_arrays

    subroutine compare_real_matrices(mat1, mat2, test_name)
        real(8), intent(in) :: mat1(:,:), mat2(:,:)
        character(len=*), intent(in) :: test_name
        integer :: i
        if (size(mat1) /= size(mat2)) then
            print *, test_name, ": FAILED (different sizes)"
            return
        end if
        if (all(abs(mat1 - mat2) < TOLERANCE)) then
            print *, test_name, ": PASSED"
        else
            print *, test_name, ": FAILED"
            print '(A)', "Matrix 1:"
            do i = 1, size(mat1,1)
                print '( *(F8.2,1X) )', mat1(i,:)
            end do
            print '(A)', "Matrix 2:"
            do i = 1, size(mat2,1)
                print '( *(F8.2,1X) )', mat2(i,:)
            end do
        end if
    end subroutine compare_real_matrices

    subroutine compare_real_scalars(val1, val2, test_name)
        real(8), intent(in) :: val1, val2
        character(len=*), intent(in) :: test_name
        if (abs(val1 - val2) < TOLERANCE) then
            print *, test_name, ": PASSED"
        else
            print *, test_name, ": FAILED"
            print *, "Value 1:", val1
            print *, "Value 2:", val2
        end if
    end subroutine compare_real_scalars

    subroutine compare_int_arrays(arr1, arr2, test_name)
        integer, intent(in) :: arr1(:), arr2(:)
        character(len=*), intent(in) :: test_name
         if (size(arr1) /= size(arr2)) then
            print *, test_name, ": FAILED (different sizes)"
            return
        end if
        if (all(arr1 == arr2)) then
            print *, test_name, ": PASSED"
        else
            print *, test_name, ": FAILED"
            print *, "Array 1:", arr1
            print *, "Array 2:", arr2
        end if
    end subroutine compare_int_arrays

    subroutine print_model_formula(nbases, mask, truncated, cov, root, coefficients, y_mean, label)
        integer, intent(in) :: nbases
        logical, intent(in) :: mask(:,:)
        logical, intent(in) :: truncated(:,:)
        integer, intent(in) :: cov(:,:)
        real(8), intent(in) :: root(:,:)
        real(8), intent(in) :: coefficients(:)
        real(8), intent(in) :: y_mean
        character(len=*), intent(in) :: label
        integer :: i, j
        character(len=256) :: term
        logical :: first_factor

        print *, trim(label)
        write(*,'(A,F8.2)') 'y = ', y_mean
        do i = 2, nbases
            if (any(mask(:,i))) then
                write(*,'(A,F8.2,A)', advance='no') ' + ', coefficients(i-1), ' * '
                first_factor = .true.
                do j = 1, size(mask,1)
                    if (mask(j,i)) then
                        if (.not. first_factor) write(*,'(A)', advance='no') ' * '
                        if (truncated(j,i)) then
                            write(term, '(A,I0,A,F6.2,A)') 'max(0, x[', cov(j,i), '] - ', root(j,i), ')'
                        else
                            write(term, '(A,I0,A,F6.2,A)') '(x[', cov(j,i), '] - ', root(j,i), ')'
                        end if
                        write(*,'(A)', advance='no') trim(term)
                        first_factor = .false.
                    end if
                end do
                print *  ! Newline after each basis
            end if
        end do
    end subroutine print_model_formula
end module utils