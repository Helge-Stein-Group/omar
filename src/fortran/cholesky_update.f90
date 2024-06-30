module cholesky_update_module
   implicit none
contains
   subroutine update_cholesky(chol, update_vectors, multipliers, n, m)
      ! Arguments
      integer, intent(in) :: n  ! Size of the Cholesky decomposition matrix
      integer, intent(in) :: m  ! Number of update vectors/multipliers
      real(8), intent(inout) :: chol(n, n)
      real(8), intent(in) :: update_vectors(n, m)
      real(8), intent(in) :: multipliers(m)

      ! Local variables
      real(8) :: diag(n)
      real(8) :: u(n, n)
      real(8) :: b(n)
      integer :: i, j, k

      ! Perform the update
      do k = 1, m
         ! Extract diagonal
         do i = 1, n
            diag(i) = chol(i, i)
         end do

         ! Divide chol by its diagonal elements (broadcast manually)
         do j = 1, n
            chol(j, :) = chol(j, :) / diag(j)
         end do

         diag = diag**2

         ! Initialize u
         u = 0.0
         u(:, 1) = update_vectors(:, k)
         u(2:, 1) = u(2:, 1) - update_vectors(1, k) * chol(1, 2:)

         b = 1.0

         ! Perform the update
         do i = 2, n
            u(:, i) = u(:, i-1)
            u(i+1:, i) = u(i+1:, i) - u(i, i-1) * chol(i, i+1:)
            b(i) = b(i-1) + multipliers(k) * u(i-1, i-1)**2 / diag(i-1)
         end do

         ! Update the Cholesky decomposition
         do i = 1, n
            chol(i, i) = sqrt(diag(i) + multipliers(k) / b(i) * u(i, i)**2)
            chol(i, i+1:) = chol(i, i+1:) * chol(i,i)
            chol(i, i+1:) = chol(i, i+1:) + multipliers(k) / b(i) * u(i, i) * u(i+1:, i) / chol(i,i)
         end do
      end do
   end subroutine update_cholesky
end module cholesky_update_module
