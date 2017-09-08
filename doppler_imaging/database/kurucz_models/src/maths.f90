module maths_mod
use vars_mod, only : PHC, PK, PHK, PI, PME, PH, PC, PH2C3, PHC2, identity, fact
implicit none

contains

! ---------------------------------------------------------
! Given x(:) and y(:) which tabulate a function and the derivative at the boundary points
! this function returns the second derivative of the spline at each point
! ---------------------------------------------------------
		subroutine splin1(x,y,yp1,ypn,y2)
		real*8, INTENT(IN) :: x(:), y(:), yp1, ypn
		real*8, INTENT(INOUT) :: y2(size(x))
		integer :: n, i, k
		real*8 :: p, qn, sig, un, u(size(x))

			n = size(x)

			if (yp1 > .99d30) then
				y2(1) = 0.d0
				u(1) = 0.d0
			else
				y2(1) = -0.5d0
				u(1) = (3.d0/(x(2)-x(1)))*((y(2)-y(1))/(x(2)-x(1))-yp1)
			endif

			do i = 2, n-1
				sig = (x(i)-x(i-1))/(x(i+1)-x(i-1))
				p = sig * y2(i-1)+2.d0
				y2(i) = (sig-1.d0)/p
				u(i) = (6.d0*((y(i+1)-y(i))/(x(i+1)-x(i))-(y(i)-y(i-1))/(x(i)-x(i-1)))/&
					(x(i+1)-x(i-1))-sig*u(i-1))/p
			enddo
			if (ypn > .99d30) then
				qn = 0.d0
				un = 0.d0
			else
				qn = 0.5d0
				un = (3.d0/(x(n)-x(n-1)))*(ypn-(y(n)-y(n-1))/(x(n)-x(n-1)))
			endif

			y2(n) = (un-qn*u(n-1))/(qn*y2(n-1)+1.d0)

			do k = n-1, 1, -1
				y2(k) = y2(k)*y2(k+1)+u(k)
			enddo

		end subroutine splin1

! ---------------------------------------------------------
! Given xa(:) and ya(:) which tabulate a function, returns the interpolation using
! splines of vector x(:) in y(:)
! ---------------------------------------------------------
		subroutine spline(xa,ya,x,y)
		real*8, INTENT(INOUT) :: y(:)
		real*8, INTENT(IN) :: xa(:), ya(:), x(:)
		real*8 :: y2a(size(xa))
		integer :: n_x, n, i, k, khi, klo
		real*8 :: a, b, h, extrap

			n = size(xa)
			n_x = size(x)
			call splin1(xa,ya,1.d30,1.d30,y2a)

			do i = 1, n_x

! Downward extrapolation
				if (x(i) < xa(1)) then
!					y(i) = ya(1)
					y(i) = ya(1) + (ya(1)-ya(2))/(xa(1)-xa(2)) * (xa(1) - x(i))
				else

! Upward extrapolation
				if (x(i) > xa(n)) then
!					y(i) = ya(n)
					y(i) = ya(n) + (ya(n)-ya(n-1)) / (xa(n)-xa(n-1)) * (x(i) - xa(n))
				else
! In range
						klo = 1
						khi = n
1						if(khi-klo > 1) then
							k = (khi+klo)/2
							if (xa(k) > x(i)) then
								khi = k
							else
								klo = k
							endif
							go to 1
						endif

						h = xa(khi)-xa(klo)

						if (h == 0.d0) then
							print *, 'bad xa input in spline'
							pause
						endif
						a = (xa(khi)-x(i))/h
						b = (x(i)-xa(klo))/h

						y(i) = a*ya(klo)+b*ya(khi)+((a**3.d0-a)*y2a(klo)+(b**3.d0-b)*y2a(khi))*(h**2.d0)/6.d0
					endif
				endif
			enddo

		end subroutine spline
		
!----------------------------------------------------------------
! This function returns the strength of the Zeeman components
! Do it using table 3.1 of Landi degl'Innocenti & Landolfi (2004)
! to avoid overflow with the factorials if using the 3-j symbol
!----------------------------------------------------------------
	function strength_zeeman(J_up,J_low,M_up,M_low)
	real(kind=8) :: J_up, J_low, M_up, M_low, strength_zeeman, strength_zeeman2

! 		strength_zeeman2 = 3.d0 * w3js(int(2.d0*J_up),int(2.d0*J_low),2,int(2.d0*M_up),&
! 					-int(2.d0*M_low),int(2.d0*(M_low-M_up)))**2

		if (J_up == J_low+1) then
			if (M_up == M_low+1) then
				strength_zeeman = (3.d0*(J_low+M_low+1)*(J_low+M_low+2)) / (2.d0*(J_low+1.d0)*(2.d0*J_low+1.d0)*(2.d0*J_low+3.d0))
 				return
			endif
			if (M_up == M_low) then
				strength_zeeman = (3.d0*(J_low-M_low+1)*(J_low+M_low+1)) / ((J_low+1.d0)*(2.d0*J_low+1.d0)*(2.d0*J_low+3.d0))
 				return
			endif
			if (M_up == M_low-1) then
				strength_zeeman = (3.d0*(J_low-M_low+1)*(J_low-M_low+2)) / (2.d0*(J_low+1.d0)*(2.d0*J_low+1.d0)*(2.d0*J_low+3.d0))
 				return
			endif
		endif

		if (J_up == J_low) then
			if (M_up == M_low+1) then
				strength_zeeman = (3.d0*(J_low-M_low)*(J_low+M_low+1)) / (2.d0*J_low*(J_low+1.d0)*(2.d0*J_low+1.d0))
 				return
			endif
			if (M_up == M_low) then
				strength_zeeman = (3.d0*M_low**2) / (J_low*(J_low+1.d0)*(2.d0*J_low+1.d0))
 				return
			endif
			if (M_up == M_low-1) then
				strength_zeeman = (3.d0*(J_low+M_low)*(J_low-M_low+1)) / (2.d0*J_low*(J_low+1.d0)*(2.d0*J_low+1.d0))
 				return
			endif
		endif

		if (J_up == J_low-1) then
			if (M_up == M_low+1) then
				strength_zeeman = (3.d0*(J_low-M_low)*(J_low-M_low-2)) / (2.d0*J_low*(2.d0*J_low-1.d0)*(2.d0*J_low+1.d0))
 				return
			endif
			if (M_up == M_low) then
				strength_zeeman = (3.d0*(J_low-M_low)*(J_low+M_low)) / (J_low*(2.d0*J_low-1.d0)*(2.d0*J_low+1.d0))
 				return
			endif
			if (M_up == M_low-1) then
				strength_zeeman = (3.d0*(J_low+M_low)*(J_low+M_low-1)) / (2.d0*J_low*(2.d0*J_low-1.d0)*(2.d0*J_low+1.d0))
 				return
			endif
		endif
					
	end function strength_zeeman

!----------------------------------------------------------------
! This function calculates the 3-j symbol
! J_i and M_i have to be twice the actual value of J and M
!----------------------------------------------------------------
   function w3js(j1,j2,j3,m1,m2,m3)
   integer :: m1, m2, m3, j1, j2, j3
	integer :: ia, ib, ic, id, ie, im, ig, ih, z, zmin, zmax, jsum
	real(kind=8) :: w3js, cc, denom, cc1, cc2

      w3js = 0.d0
      if (m1+m2+m3 /= 0) return
      ia = j1 + j2
      if (j3 > ia) return
      ib = j1 - j2
      if (j3 < abs(ib)) return
		if (abs(m1) > j1) return
		if (abs(m2) > j2) return
		if (abs(m3) > j3) return

      jsum = j3 + ia
      ic = j1 - m1
      id = j2 - m2

		ie = j3 - j2 + m1
		im = j3 - j1 - m2
		zmin = max0(0,-ie,-im)
		ig = ia - j3
		ih = j2 + m2
		zmax = min0(ig,ih,ic)
		cc = 0.d0

		do z = zmin, zmax, 2
			denom = fact(z/2)*fact((ig-z)/2)*fact((ic-z)/2)*fact((ih-z)/2)*&
				fact((ie+z)/2)*fact((im+z)/2)
      	if (mod(z,4) /= 0) denom = -denom
			cc = cc + 1.d0/denom
		enddo

		cc1 = fact(ig/2)*fact((j3+ib)/2)*fact((j3-ib)/2)/fact((jsum+2)/2)
      cc2 = fact((j1+m1)/2)*fact(ic/2)*fact(ih/2)*fact(id/2)*fact((j3-m3)/2)*fact((j3+m3)/2)
      cc = cc * sqrt(1.d0*cc1*cc2)
		if (mod(ib-m3,4) /= 0) cc = -cc
		w3js = cc
		if (abs(w3js) < 1.d-8) w3js = 0.d0
1000 		return
   end function w3js

!----------------------------------------------------------------
! Calculates the factorial of the integers up to 301 and save it into the fact array
!----------------------------------------------------------------
   subroutine factrl
	integer :: i
      fact(0) = 1.d0
      do i = 1,101
			fact(i) = fact(i-1) * dble(i)
		enddo
   end subroutine factrl

! ---------------------------------------------------------
! Returns Planck's function for a frequency and a set of temperatures in cgs
! INPUT:
!     - nu : frequency
!     - T : temperature
! ---------------------------------------------------------
   function planck_vect(nu,T)
	real*8, INTENT(IN) :: nu, T(:)
   real*8 :: planck_vect(size(T))
   real*8 :: temporal(size(T))

		where (T /= 0.d0)
			planck_vect = PHC * nu**3.d0 / (dexp(PHK * nu / T) - 1.d0)
		elsewhere
			planck_vect = 0.d0
		endwhere

   end function planck_vect

! ---------------------------------------------------------
! Returns Planck's function for a frequency and temperature in cgs
! INPUT:
!     - nu : frequency (Hz)
!     - T : temperature (K)
! ---------------------------------------------------------
   function planck(nu,T)
   real*8 :: planck
   real*8, INTENT(IN) :: nu, T
   real*8 :: temporal
      if (T /= 0.d0) then
         planck = PHC * nu**3.d0 / (dexp(PHK * nu / T) - 1.d0)
      else
			planck = 0.d0
      endif

   end function planck

! ---------------------------------------------------------
! Returns Planck's function for a wavelength and a set of temperatures in cgs
! INPUT:
!     - lambda : wavelength
!     - T : temperature
! ---------------------------------------------------------
   function planck_vect_lambda(lambda,T)
	real*8, INTENT(IN) :: lambda, T(:)
   real*8 :: planck_vect_lambda(size(T))
   real*8 :: temporal(size(T))

		where (T /= 0.d0)
			planck_vect_lambda = PHC2 / lambda**5.d0 / (dexp(PHK * PC / (lambda * T)) - 1.d0)
		elsewhere
			planck_vect_lambda = 0.d0
		endwhere

   end function planck_vect_lambda

! ---------------------------------------------------------
! Returns Planck's function for a wavelength and temperature in cgs
! INPUT:
!     - lambda : wavelength (cm)
!     - T : temperature (K)
! ---------------------------------------------------------
   function planck_lambda(lambda,T)
   real*8 :: planck_lambda
   real*8, INTENT(IN) :: lambda, T
   real*8 :: temporal

      if (T /= 0.d0) then
         temporal = dexp(PHK * PC / (lambda * T) - 1.d0)
         planck_lambda = PHC2 / lambda**5.d0 / temporal
      else
			planck_lambda = 0.d0
      endif

   end function planck_lambda

! ---------------------------------------------------------
! Returns the derivative of the Planck's function for a wavelength and a set of temperatures in cgs
! INPUT:
!     - lambda : wavelength
!     - T : temperature
! ---------------------------------------------------------
   function planck_vect_lambda_deriv(lambda,T)
	real*8, INTENT(IN) :: lambda, T(:)
   real*8 :: planck_vect_lambda_deriv(size(T))
   real*8 :: temporal(size(T))

		where (T /= 0.d0)
			temporal = dexp(PHK * PC / (lambda * T))
			planck_vect_lambda_deriv = PH2C3 / (PK*T**2*lambda**6.d0) / (temporal-1.d0) * temporal
		elsewhere
			planck_vect_lambda_deriv = 0.d0
		endwhere

   end function planck_vect_lambda_deriv

! ---------------------------------------------------------
! Returns the derivative of the Planck's function for a wavelength and temperature in cgs
! INPUT:
!     - lambda : wavelength (cm)
!     - T : temperature (K)
! ---------------------------------------------------------
   function planck_lambda_deriv(lambda,T)
   real*8 :: planck_lambda_deriv
   real*8, INTENT(IN) :: lambda, T
   real*8 :: temporal

      if (T /= 0.d0) then
         temporal = dexp(PHK * PC / (lambda * T))
         planck_lambda_deriv = PH2C3 / (PK*T**2*lambda**6.d0) / (temporal-1.d0) * temporal
      else
			planck_lambda_deriv = 0.d0
      endif

   end function planck_lambda_deriv

!-------------------------------------------------------------------
! Returns which is the index of the array wave_total closer to wave
!-------------------------------------------------------------------
	function close_to(wave_total, wave)
	real(kind=8) :: wave_total(:), wave
	real(kind=8), allocatable :: diff(:)
	integer :: n, i, location(1)
	integer :: close_to

		n = size(wave_total)
		allocate(diff(n))
		diff = dabs(wave_total-wave)
		location = minloc(diff)
		deallocate(diff)
		close_to = location(1)

	end function close_to

!-------------------------------------------------------------------
! Returns which is the index of the array wave_total closer to wave
!-------------------------------------------------------------------
	function extremes(freq, line_frequency, delta)
	real(kind=8) :: freq(:), line_frequency, delta
	integer :: left, right, extremes(2)


		left = close_to(freq, line_frequency-delta)
		right = close_to(freq, line_frequency+delta)

		extremes(1) = min(left,right)
		extremes(2) = max(left,right)

	end function extremes

!-----------------------------------------------------------------
! Calculates the ratio between two ionization stages using the Saha equation
!-----------------------------------------------------------------
	function saha(t, pe, ul ,uu, chi)
	real(kind=8) :: t(:), pe(:), ul(:), uu(:), chi, saha(size(t))
	real(kind=8) :: constant

		constant = 2.d0 * (2.d0*PI*PME)**(1.5d0) / PH**3 * PK**(2.5d0)
		saha = constant * uu / ul * t**2.5d0 * 10.d0**(-5040.d0*chi/t) / pe

	end function saha

!--------------------------------------------------------------
! Generates Voigt and anomalous dispersion profiles using a different damping for each frequency
! point
! See Humlicek (1982) JQSRT 27, 437
!--------------------------------------------------------------
	function fvoigt(da, dv)
	real*8 :: da
	real*8 :: dv, fvoigt
	complex :: w4, z, t, u, v4
	real*8 :: s
	integer :: i, n

! There is a problem for very small dampings. We force it to be 0 using a threshold
		if (da < 1.d-3) da = 0.d0
		z = cmplx(dv, da)
		t = cmplx(da, -dv)
		s = dabs(dv) + da
		u = t*t


		if (s >= 15.d0) then
			w4 = t * 0.5641896d0 / (0.5d0+u)
		elseif (s >= 5.5) then
			w4 = t*(1.410474d0+u*0.5641896d0)/(0.75d0+u*(3.d0+u))
		elseif (da >= 0.195d0*dabs(dv)-0.176d0) then
			w4 = (16.4955d0+t*(20.20933d0+t*(11.96482d0+t*(3.778987d0+t*0.5642236d0)))) / &
				(16.4955d0+t*(38.82363d0+t*(39.27121d0+t*(21.69274d0+t*(6.699398d0+t)))))
		else
			w4 = t*(36183.31d0-u*(3321.9905d0-u*(1540.787d0-u*(219.0313d0-u*(35.76683d0-&
				u*(1.320522d0-u*0.56419d0))))))
			v4 = (32066.6d0-u*(24322.84d0-u*(9022.228d0-u*(2186.181d0-u*(364.2191d0-&
				u*(61.57037d0-u*(1.841439d0-u)))))))
			w4 = cexp(u) - w4/v4
		endif
		
		fvoigt = dble(w4)

	end function fvoigt

!--------------------------------------------------------------
! Generates Voigt and anomalous dispersion profiles using a different damping for each frequency
! point
! See Humlicek (1982) JQSRT 27, 437
!--------------------------------------------------------------
	function vecfvoigt(da, dv)
	real*8 :: da(:)
	real*8 :: dv(:), vecfvoigt(size(dv))
	complex :: w4, z, t, u, v4
	real*8 :: s
	integer :: i, n

		n = size(dv)
		do i = 1, n
! There is a problem for very small dampings. We force it to be 0 using a threshold
			if (da(i) < 1.d-3) da(i) = 0.d0
			z = cmplx(dv(i), da(i))
			t = cmplx(da(i), -dv(i))
			s = dabs(dv(i)) + da(i)
			u = t*t


			if (s >= 15.d0) then
				w4 = t * 0.5641896d0 / (0.5d0+u)
			elseif (s >= 5.5) then
				w4 = t*(1.410474d0+u*0.5641896d0)/(0.75d0+u*(3.d0+u))
			elseif (da(i) >= 0.195d0*dabs(dv(i))-0.176d0) then
				w4 = (16.4955d0+t*(20.20933d0+t*(11.96482d0+t*(3.778987d0+t*0.5642236d0)))) / &
					(16.4955d0+t*(38.82363d0+t*(39.27121d0+t*(21.69274d0+t*(6.699398d0+t)))))
			else
				w4 = t*(36183.31d0-u*(3321.9905d0-u*(1540.787d0-u*(219.0313d0-u*(35.76683d0-&
					u*(1.320522d0-u*0.56419d0))))))
				v4 = (32066.6d0-u*(24322.84d0-u*(9022.228d0-u*(2186.181d0-u*(364.2191d0-&
					u*(61.57037d0-u*(1.841439d0-u)))))))
				w4 = cexp(u) - w4/v4
			endif
			vecfvoigt(i) = dble(w4)
		enddo

	end function vecfvoigt

!--------------------------------------------------------------
! Generates Voigt and anomalous dispersion profiles using a different damping for each frequency
! point
! See Humlicek (1982) JQSRT 27, 437
!--------------------------------------------------------------
	function vecfvoigt_zeeman(da, dv)
	real*8 :: da(:)
	real*8 :: dv(:), vecfvoigt_zeeman(2,size(dv))
	complex :: w4, z, t, u, v4
	real*8 :: s
	integer :: i, n

		n = size(dv)
		do i = 1, n
! There is a problem for very small dampings. We force it to be 0 using a threshold
			if (da(i) < 1.d-3) da(i) = 0.d0
			z = cmplx(dv(i), da(i))
			t = cmplx(da(i), -dv(i))
			s = dabs(dv(i)) + da(i)
			u = t*t


			if (s >= 15.d0) then
				w4 = t * 0.5641896d0 / (0.5d0+u)
			elseif (s >= 5.5) then
				w4 = t*(1.410474d0+u*0.5641896d0)/(0.75d0+u*(3.d0+u))
			elseif (da(i) >= 0.195d0*dabs(dv(i))-0.176d0) then
				w4 = (16.4955d0+t*(20.20933d0+t*(11.96482d0+t*(3.778987d0+t*0.5642236d0)))) / &
					(16.4955d0+t*(38.82363d0+t*(39.27121d0+t*(21.69274d0+t*(6.699398d0+t)))))
			else
				w4 = t*(36183.31d0-u*(3321.9905d0-u*(1540.787d0-u*(219.0313d0-u*(35.76683d0-&
					u*(1.320522d0-u*0.56419d0))))))
				v4 = (32066.6d0-u*(24322.84d0-u*(9022.228d0-u*(2186.181d0-u*(364.2191d0-&
					u*(61.57037d0-u*(1.841439d0-u)))))))
				w4 = cexp(u) - w4/v4
			endif
			vecfvoigt_zeeman(1,i) = dble(w4)
			vecfvoigt_zeeman(2,i) = aimag(w4)
		enddo

	end function vecfvoigt_zeeman


!--------------------------------------------------------------
! Inversion of a 4x4 matrix
!--------------------------------------------------------------
	subroutine invert(a)
	real(kind=8) :: a(4,4)
	real(kind=8) :: b(4,4), det, maxim, fabsmax
	! First some tests of singularity
		b = dabs(a)
		maxim = maxval(b)
		fabsmax = 1.d0 / maxim
		if (maxim == 0.d0) then
			print *, 'Singularity in the inversion'
!			stop
		endif

		a = a * fabsmax

   	b(1,1) = a(2,2) * a(3,3) * a(4,4) + a(2,3) * a(3,4) * a(4,2)&
      	+ a(2,4) * a(3,2) * a(4,3) - a(2,2) * a(3,4) * a(4,3)&
	   	- a(2,3) * a(3,2) * a(4,4) - a(2,4) * a(3,3) * a(4,2)
   	b(2,1) = a(2,3) * a(3,1) * a(4,4) + a(2,4) * a(3,3) * a(4,1)&
	   	+ a(2,1) * a(3,4) * a(4,3) - a(2,3) * a(3,4) * a(4,1)&
   		- a(2,4) * a(3,1) * a(4,3) - a(2,1) * a(3,3) * a(4,4)
   	b(3,1) = a(2,4) * a(3,1) * a(4,2) + a(2,1) * a(3,2) * a(4,4)&
	   	+ a(2,2) * a(3,4) * a(4,1) - a(2,4) * a(3,2) * a(4,1)&
   		- a(2,1) * a(3,4) * a(4,2) - a(2,2) * a(3,1) * a(4,4)
   	b(4,1) = a(2,1) * a(3,3) * a(4,2) + a(2,2) * a(3,1) * a(4,3)&
	   	+ a(2,3) * a(3,2) * a(4,1) - a(2,1) * a(3,2) * a(4,3)&
   		- a(2,2) * a(3,3) * a(4,1) - a(2,3) * a(3,1) * a(4,2)
   	b(1,2) = a(3,2) * a(4,4) * a(1,3) + a(3,3) * a(4,2) * a(1,4)&
	   	+ a(3,4) * a(4,3) * a(1,2) - a(3,2) * a(4,3) * a(1,4)&
   		- a(3,3) * a(4,4) * a(1,2) - a(3,4) * a(4,2) * a(1,3)
   	b(2,2) = a(3,3) * a(4,4) * a(1,1) + a(3,4) * a(4,1) * a(1,3)&
	   	+ a(3,1) * a(4,3) * a(1,4) - a(3,3) * a(4,1) * a(1,4)&
   		- a(3,4) * a(4,3) * a(1,1) - a(3,1) * a(4,4) * a(1,3)
   	b(3,2) = a(3,4) * a(4,2) * a(1,1) + a(3,1) * a(4,4) * a(1,2)&
	   	+ a(3,2) * a(4,1) * a(1,4) - a(3,4) * a(4,1) * a(1,2)&
   		- a(3,1) * a(4,2) * a(1,4) - a(3,2) * a(4,4) * a(1,1)
   	b(4,2) = a(3,1) * a(4,2) * a(1,3) + a(3,2) * a(4,3) * a(1,1)&
	   	+ a(3,3) * a(4,1) * a(1,2) - a(3,1) * a(4,3) * a(1,2)&
   		- a(3,2) * a(4,1) * a(1,3) - a(3,3) * a(4,2) * a(1,1)
   	b(1,3) = a(4,2) * a(1,3) * a(2,4) + a(4,3) * a(1,4) * a(2,2)&
   		+ a(4,4) * a(1,2) * a(2,3) - a(4,2) * a(1,4) * a(2,3)&
	   	- a(4,3) * a(1,2) * a(2,4) - a(4,4) * a(1,3) * a(2,2)
   	b(2,3) = a(4,3) * a(1,1) * a(2,4) + a(4,4) * a(1,3) * a(2,1)&
	   	+ a(4,1) * a(1,4) * a(2,3) - a(4,3) * a(1,4) * a(2,1)&
   		- a(4,4) * a(1,1) * a(2,3) - a(4,1) * a(1,3) * a(2,4)
   	b(3,3) = a(4,4) * a(1,1) * a(2,2) + a(4,1) * a(1,2) * a(2,4)&
	   	+ a(4,2) * a(1,4) * a(2,1) - a(4,4) * a(1,2) * a(2,1)&
   		- a(4,1) * a(1,4) * a(2,2) - a(4,2) * a(1,1) * a(2,4)
   	b(4,3) = a(4,1) * a(1,3) * a(2,2) + a(4,2) * a(1,1) * a(2,3)&
	   	+ a(4,3) * a(1,2) * a(2,1) - a(4,1) * a(1,2) * a(2,3)&
   		- a(4,2) * a(1,3) * a(2,1) - a(4,3) * a(1,1) * a(2,2)
   	b(1,4) = a(1,2) * a(2,4) * a(3,3) + a(1,3) * a(2,2) * a(3,4)&
	   	+ a(1,4) * a(2,3) * a(3,2) - a(1,2) * a(2,3) * a(3,4)&
   		- a(1,3) * a(2,4) * a(3,2) - a(1,4) * a(2,2) * a(3,3)
   	b(2,4) = a(1,3) * a(2,4) * a(3,1) + a(1,4) * a(2,1) * a(3,3)&
	   	+ a(1,1) * a(2,3) * a(3,4) - a(1,3) * a(2,1) * a(3,4)&
   		- a(1,4) * a(2,3) * a(3,1) - a(1,1) * a(2,4) * a(3,3)
   	b(3,4) = a(1,4) * a(2,2) * a(3,1) + a(1,1) * a(2,4) * a(3,2)&
	   	+ a(1,2) * a(2,1) * a(3,4) - a(1,4) * a(2,1) * a(3,2)&
   		- a(1,1) * a(2,2) * a(3,4) - a(1,2) * a(2,4) * a(3,1)
   	b(4,4) = a(1,1) * a(2,2) * a(3,3) + a(1,2) * a(2,3) * a(3,1)&
	   	+ a(1,3) * a(2,1) * a(3,2) - a(1,1) * a(2,3) * a(3,2)&
   		- a(1,2) * a(2,1) * a(3,3) - a(1,3) * a(2,2) * a(3,1)

		det = a(1,1) * b(1,1) + a(1,2) * b(2,1) + a(1,3) * b(3,1) + a(1,4) * b(4,1)

		a = b * (fabsmax / det)

	end subroutine invert

!----------------------------------------------------------------
! Given the values of the Zeeman opacities, fill the absorption matrix
! for each depth point
!----------------------------------------------------------------
		subroutine fill_matrix(opacity,ab_matrix)
		real(kind=8) :: opacity(:,:), ab_matrix(:,:,:)

			ab_matrix(1,1,:) = opacity(1,:)   !eta_I
			ab_matrix(2,2,:) = opacity(1,:)   !eta_I
			ab_matrix(3,3,:) = opacity(1,:)   !eta_I
			ab_matrix(4,4,:) = opacity(1,:)   !eta_I
			ab_matrix(1,2,:) = opacity(2,:)   !eta_Q
			ab_matrix(2,1,:) = opacity(2,:)   !eta_Q
			ab_matrix(1,3,:) = opacity(3,:)   !eta_U
			ab_matrix(3,1,:) = opacity(3,:)   !eta_U
			ab_matrix(1,4,:) = opacity(4,:)   !eta_V
			ab_matrix(4,1,:) = opacity(4,:)   !eta_V
			ab_matrix(2,3,:) = opacity(7,:)   !rho_V
			ab_matrix(3,2,:) = -opacity(7,:)  !-rho_V
			ab_matrix(2,4,:) = -opacity(6,:)  !-rho_U
			ab_matrix(4,2,:) = opacity(6,:)   !rho_U
			ab_matrix(3,4,:) = opacity(5,:)   !rho_Q
			ab_matrix(4,3,:) = -opacity(5,:)  !-rho_Q

		end subroutine fill_matrix

! ---------------------------------------------------------
! Solves RTE using short-characteristics. Source function is supposed to vary parabolically
! between points M, O and P and RTE is integrated analitically, resulting in
!            I(O)=I(M)*exp(-tau(MO))+psi(M)*S(M)+psi(O)*S(O)+psi(P)*S(P)
! where psi are functions of the optical distance tau(MO) and tau(OP)
! It returns the value of the three psi coefficients
! ---------------------------------------------------------
   subroutine par_sc(dtm, dtp, psim, psi0, psip)
   real(kind=8) :: short_car_parab
   real(kind=8), INTENT(IN) :: dtm, dtp
	real(kind=8), INTENT(INOUT) :: psim, psi0, psip
   real(kind=8) :: exu, u0, u1 ,u2, d2, d3, d4

         if (dtm.ge.1.d-4) then
            exu=dexp(-dtm)
            u0=1.d0-exu
            u1=dtm-1.d0+exu
            u2=dtm**2-2.d0*dtm+2.d0-2.d0*exu
         else
            d2=dtm**2
            d3=dtm**3
            d4=dtm**4
            u0=dtm-(d2/2.d0)
            u1=(d2/2.d0)-(d3/6.d0)
            u2=(d3/3.d0)-(d4/12.d0)
        endif

        if (dtm * dtp /= 0.d0 .and. dtm**2 /= 0.d0 .and. dtp**2 /= 0.d0) then
			  psim=(u2-u1*(dtp+2.d0*dtm))/(dtm*(dtm+dtp))+u0
      	  psi0=(u1*(dtm+dtp)-u2)/(dtm*dtp)
      	  psip=(u2-dtm*u1)/(dtp*(dtm+dtp))
	 	  else
		  	  psim = 0.d0
			  psi0 = 0.d0
			  psip = 0.d0
		  endif

   end subroutine par_sc

! ---------------------------------------------------------
! Solves RTE using short-characteristics. Source function is supposed to vary parabolically
! between points M, O and P and RTE is integrated analitically, resulting in
!            I(O)=I(M)*exp(-tau(MO))+psi(M)*S(M)+psi(O)*S(O)+psi(P)*S(P)
! where psi are functions of the optical distance tau(MO) and tau(OP)
! It returns the value of the three psi coefficients
! Works for vectorial functions
! ---------------------------------------------------------
   subroutine par_sc_vect(dtm, dtp, psim, psi0, psip)
   real(kind=8), INTENT(IN) :: dtm(:), dtp(:)
	real(kind=8), INTENT(INOUT) :: psim(size(dtm)), psi0(size(dtm)), psip(size(dtm))
   real(kind=8) :: exu(size(dtm)), u0(size(dtm)), u1(size(dtm)) ,u2(size(dtm)), d2(size(dtm))
	real(kind=8) :: d3(size(dtm)), d4(size(dtm))


	where (dtm >= 1.d-4)
		exu = dexp(-dtm)
		u0=1.d0-exu
		u1=dtm-1.d0+exu
		u2=dtm**2-2.d0*dtm+2.d0-2.d0*exu
	elsewhere
		d2=dtm**2
		d3=dtm**3
		d4=dtm**4
		u0=dtm-(d2/2.d0)
		u1=(d2/2.d0)-(d3/6.d0)
		u2=(d3/3.d0)-(d4/12.d0)
	endwhere

	psim = 0.d0
	psip = 0.d0
	psi0 = 0.d0

	where(dtm * dtp /= 0.d0 .and. dtm**2 /= 0.d0 .and. dtp**2 /= 0.d0)
		psim=(u2-u1*(dtp+2.d0*dtm))/(dtm*(dtm+dtp))+u0
		psi0=(u1*(dtm+dtp)-u2)/(dtm*dtp)
		psip=(u2-dtm*u1)/(dtp*(dtm+dtp))
	endwhere

   end subroutine par_sc_vect

! ---------------------------------------------------------
! Solves RTE using short-characteristics. Source function is supposed to vary linearly
! between points M and O and RTE is integrated analitically, resulting in
!            I(O)=I(M)*exp(-tau(MO))+psi(M)*S(M)+psi(O)*S(O)
! where psi are functions of the optical distance tau(MO)
! It returns the value of the two psi coefficients
! ---------------------------------------------------------
   subroutine lin_sc(dtm, psim, psi0)
   real(kind=8) :: short_car_linea
   real(kind=8), INTENT(IN) :: dtm
	real(kind=8), INTENT(INOUT) :: psim, psi0
   real(kind=8) :: exu, u0, u1, c0, cm, d2

      if (dtm.ge.1.d-4) then
         exu=dexp(-dtm)
         u0=1.d0-exu
         u1=dtm-1.d0+exu

         c0=u1/dtm
         cm=u0-c0
      else
         d2=dtm**2.d0
         c0=(dtm/2.d0)-(d2/6.d0)
         cm=(dtm/2.d0)-(d2/3.d0)
      endif
		psi0 = c0
		psim = cm

   end subroutine lin_sc

! ---------------------------------------------------------
! Solves RTE using short-characteristics. Source function is supposed to vary linearly
! between points M and O and RTE is integrated analitically, resulting in
!            I(O)=I(M)*exp(-tau(MO))+psi(M)*S(M)+psi(O)*S(O)
! where psi are functions of the optical distance tau(MO)
! It returns the value of the two psi coefficients
! Works for vectorial functions
! ---------------------------------------------------------
   subroutine lin_sc_vect(dtm, psim, psi0)
   real(kind=8), INTENT(IN) :: dtm(:)
	real(kind=8), INTENT(INOUT) :: psim(size(dtm)), psi0(size(dtm))
   real(kind=8) :: exu(size(dtm)), u0(size(dtm)), u1(size(dtm)), c0(size(dtm)), cm(size(dtm)), d2(size(dtm))

	where (dtm >= 1.d-4)
		exu=dexp(-dtm)
		u0=1.d0-exu
		u1=dtm-1.d0+exu
		c0=u1/dtm
		cm=u0-c0
	elsewhere
		d2=dtm**2.d0
		c0=(dtm/2.d0)-(d2/6.d0)
		cm=(dtm/2.d0)-(d2/3.d0)
	endwhere

	psi0 = c0
	psim = cm

   end subroutine lin_sc_vect

! ---------------------------------------------------------
! Performs the formal solution of the RT equation for a plane-parallel atmosphere
! for a given source function and opacity
! Opacity and source function have the index as
! opacity(depth,frequency)
! ---------------------------------------------------------
	function formal_sol_vect(height, opacity, source, mu, boundary, idir)
	real(kind=8) :: height(:), opacity(:,:), source(:,:), boundary(:), mu
	real(kind=8) :: formal_sol_vect(size(boundary))
	integer :: k, km, kp, idir, k0, kf, ndepths, i
	real(kind=8), dimension(size(boundary)) :: chim, chi0, chip, sm, s0, sp, Inten, dtp, dtm, exu
	real(kind=8), dimension(size(boundary)) :: psim, psi0, psip
	real(kind=8) :: dm, dp

		ndepths = size(height)
		
! Boundary condition
		if (idir == 1) then
			k0 = 2
			kf = ndepths
			Inten = boundary
		endif
		if (idir == -1) then
			k0 = ndepths-1
			kf = 1
			Inten = boundary
		endif

		do k = k0, kf, idir

! Parabolic short-characteristics
			if (k /= kf) then
				km = k - idir
				kp = k + idir
				chim = opacity(km,:)
				chi0 = opacity(k,:)
				chip = opacity(kp,:)
				sm = source(km,:)
				s0 = source(k,:)
				sp = source(kp,:)
				dm = dabs(1.d0/mu*(height(k) - height(km)))
				dp = dabs(1.d0/mu*(height(kp) - height(k)))
			else
! Linear short-characteristics
				km = k - idir
				chim = opacity(km,:)
				chi0 = opacity(k,:)
				chip = 0.d0
				sm = source(km,:)
				s0 = source(k,:)
				sp = 0.d0
				dm = dabs(1.d0/mu*(height(k) - height(km)))
				dp = 0.d0
			endif

			where(chim <= 0.d0)
				chim = 1.d-15
			endwhere
			where(chi0 <= 0.d0)
				chi0 = 1.d-15
			endwhere
			where(chip <= 0.d0)
				chip = 1.d-15
			endwhere
			dtm = 0.5d0 * (chim + chi0) * dm
			dtp = 0.5d0 * (chi0 + chip) * dp

			! do i = 1, size(dtm)
			! 	print *, dtm(i)
			! enddo

			where (dtm >= 1.d-4)
				exu = dexp(-dtm)
			elsewhere
				exu = 1.d0 - dtm + 0.5d0 * dtm**2.d0
			endwhere

			if (k /= kf) then
				call par_sc_vect(dtm,dtp,psim,psi0,psip)
				Inten = Inten * exu + psim*sm + psi0*s0 + psip*sp
			else
				call lin_sc_vect(dtm,psim,psi0)
				Inten = Inten * exu + psim*sm + psi0*s0
			endif

		enddo

		formal_sol_vect = Inten

	end function formal_sol_vect

! ---------------------------------------------------------
! Performs the formal solution of the RT equation for a plane-parallel magnetized atmosphere
! for a given source function and opacity
! ---------------------------------------------------------
	function formal_sol_polarized(height, opacity, source, mu, boundary, idir)
	real(kind=8) :: height(:), opacity(:,:), source(:), mu, boundary(:)
	real(kind=8) :: formal_sol_polarized(4), Inten(4)
	integer :: k, km, kp, idir, k0, kf
	real(kind=8) :: chim, chi0, chip, dtp, dtm, exu
	real(kind=8) :: psim, psi0, psip, psim_lin, psi0_lin, dm, dp

	integer :: i, j, n
	real(kind=8), allocatable :: ab_matrix(:,:,:), source_vector(:,:)
	real(kind=8) :: sm(4), s0(4), sp(4), mat1(4,4), mat2(4,4), mat3(4,4)

		n = size(height)

		allocate(ab_matrix(4,4,n))
		allocate(source_vector(4,n))

		call fill_matrix(opacity,ab_matrix)

! Transform K into K* and then into K'
		do i = 1, 4
			do j = 1, 4
				ab_matrix(i,j,:) = ab_matrix(i,j,:) / opacity(1,:)
			enddo
			ab_matrix(i,i,:) = ab_matrix(i,i,:) - 1.d0
			source_vector(i,:) = opacity(i,:) * source / opacity(1,:)
		enddo

! Boundary condition
		if (idir == 1) then
			k0 = 2
			kf = n
			Inten = boundary
		endif
		if (idir == -1) then
			k0 = n-1
			kf = 1
			Inten = boundary
		endif

		do k = k0, kf, idir

! Parabolic short-characteristics
			if (k /= kf) then
				km = k - idir
				kp = k + idir
				chim = opacity(1,km)
				chi0 = opacity(1,k)
				chip = opacity(1,kp)
				sm = source_vector(:,km)
				s0 = source_vector(:,k)
				sp = source_vector(:,kp)
				dm = dabs(1.d0/mu*(height(k) - height(km)))
				dp = dabs(1.d0/mu*(height(kp) - height(k)))
			else
! Linear short-characteristics
				km = k - idir
				chim = opacity(1,km)
				chi0 = opacity(1,k)
				chip = 0.d0
				sm = source_vector(:,km)
				s0 = source_vector(:,k)
				sp = 0.d0
				dm = dabs(1.d0/mu*(height(k) - height(km)))
				dp = 0.d0
			endif

			dtm = 0.5d0 * (chim + chi0) * dm
			dtp = 0.5d0 * (chi0 + chip) * dp

			if (dtm >= 1.d-4) then
				exu = dexp(-dtm)
			else
				exu = 1.d0 - dtm + 0.5d0 * dtm**2.d0
			endif

			call lin_sc(dtm,psim_lin,psi0_lin)
			mat1 = exu * identity - psim_lin*ab_matrix(:,:,km)
			mat2 = identity + psi0_lin * ab_matrix(:,:,k)
			call invert(mat2)

			if (k /= kf) then
				call par_sc(dtm,dtp,psim,psi0,psip)
				Inten = matmul(mat2,matmul(mat1,Inten) + psim*sm + psi0*s0 + psip*sp)
			else
				call lin_sc(dtm,psim,psi0)
				Inten = matmul(mat2,matmul(mat1,Inten) + psim*sm + psi0*s0)
			endif

		enddo

		deallocate(ab_matrix)
		deallocate(source_vector)

		formal_sol_polarized = Inten

	end function formal_sol_polarized

end module maths_mod