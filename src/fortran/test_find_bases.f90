program test_find_bases
    use omars
    implicit none

    ! Declare variables
    real(8), allocatable :: x(:,:), y(:)
    real(8) :: y_mean
    integer :: max_nbases, max_ncandidates, nbases
    real(8) :: aging_factor
    integer :: smoothness
    integer, allocatable :: covariates(:,:)
    real(8) ,allocatable ::  nodes(:,:)
    logical, allocatable :: hinges(:,:), where(:,:)
    real(8), allocatable :: coefficients(:)

    ! Initialize variables (example values, replace with actual data)
    max_nbases = 11
    max_ncandidates = 5
    aging_factor = 0.0
    smoothness = 3

    ! Allocate and initialize x and y with example data
    allocate(x(100, 2))
    allocate(y(100))
    ! Fill x and y with example data
    call random_number(x)
    call random_number(y)
    y_mean = sum(y) / size(y)

    ! Allocate output variables
    allocate(covariates(max_nbases, max_nbases))
    allocate(nodes(max_nbases, max_nbases))
    allocate(hinges(max_nbases, max_nbases))
    allocate(where(max_nbases, max_nbases))
    allocate(coefficients(max_nbases - 1))

    ! Call the find_bases subroutine
    call find_bases(x, y, y_mean, max_nbases, max_ncandidates, aging_factor, smoothness, &
                    nbases, covariates, nodes, hinges, where, coefficients)

    ! Print results for debugging
    print *, 'Number of bases:', nbases
    print *, 'Coefficients:', coefficients

end program test_find_bases