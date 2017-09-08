module atomic_mod
use types_mod, only : spectrum_type, atmosphere_type, config_type, atomic_transition_type
use vars_mod, only : PI, PE, PC, PK, PH, UMA, OPA, PHK, ionization_state, EV_ERG, SQRTPI, conf, LARMOR
use maths_mod, only : extremes, saha, vecfvoigt, planck_vect_lambda, planck_lambda, formal_sol_vect, formal_sol_polarized,&
	strength_zeeman, vecfvoigt_zeeman, fvoigt
use background_opacity_mod, only : background_opacity
implicit none

contains

!-----------------------------------------------------------------
! Calculate the damping constant a for the Voigt profile
! following Gray 1976 "The observation and analysis of stellar photospheres"
!  Natural damping: eq (11.19)
!  van der Waals damping: eq (11.35) & (11.36)
!  Stark damping: eq (11.35) & (11.36)
!  Damping: eq (11.69)
! ion_pot : ionization potential in eV
! exc_pot_lower : excitation potential of the lower level in eV
! T : temperature array
! Pg : gas pressure array
! lambda : wavelength of the transition in A
! doppler_width : Doppler width in cm/s
!-----------------------------------------------------------------
	subroutine calculate_damping(T, nHtot, Pe, Pg, doppler_width, ion_pot, exc_pot_lower, gamma_rad, gamma_stark, gamma_vdw, lambda, &
		damping_out)
	real(kind=8) :: ion_pot, exc_pot_lower, T(:), nHtot(:), Pe(:), Pg(:), lambda, doppler_width(:)
	real(kind=8) :: damping_out(size(T)), gamma_rad, gamma_stark, gamma_vdw
	integer :: n, trans
	real(kind=8) :: c6, chi_lambda, cte, a, b
	real(kind=8), allocatable :: gamma6(:), gamma_nat(:), gamma_s(:)

		n = size(T)
		
		allocate(gamma6(n))
		allocate(gamma_s(n))
		allocate(gamma_nat(n))


! If the damping is not in the linelist, then use the formulas. In the other case, use those
! given by Kurucz

! RADIATIVE DAMPING
! Radiative damping
! 			Reference: Gray 1976, "The observation and analysis of stellar
!  		photospheres", pag 227, just after Eq. (11-19). This approximation
!  		is poor, but the radiative damping is usually negligible anyway.		

		gamma_nat = 0.22233d0 / (lambda*1.d-8)**2	

! STARK DAMPING
! Stark damping
! Formula used: gamma_4=38.8*(v**(1./3.))*(C4**(2./3.))*N , from
!   Unsold 1955, "Physik der Sternatmospharen", 2nd ed.,
!   Springer-Verlag, pp. 326 and 331. According to Gray (ref above),
!   pag 237, this is similar to
!   log (gamma_4) = 19.4 + 2./3*log C4 + log Pe - 5./6.*log T
!   The value of log C4 is approximated by -14. (see Gray, Table 11-3)

		gamma_s = 19.4 + 2.d0/3.d0*(-14.d0) + dlog10(Pe) - 5.d0/6.d0*dlog10(T)
		gamma_s = 10.d0**(gamma_s)

! HYDROGEN DAMPING
! Van der Waals damping:
! Reference: Gray (see above, pag 239, Eqs. (11-35) and (11-36))
! Formula used:
!    log (gamma_vdw) = 19.6 + 2./5.*log (C6(H)) + log Pg - 7./10.*log T

		chi_lambda = (PH*PC/EV_ERG) / (lambda*1.d-8)
		a = ion_pot-exc_pot_lower-chi_lambda
		b = ion_pot-exc_pot_lower
		c6 = 0.3d-30 * (1.d0/a**2 - 1.d0/b**2)
! It seems that the ionization potential and the energy of some levels are inconsistent, and E_exc>I
		if (c6 > 0.d0) then
			gamma6 = 10.d0**(19.6d0 + 0.4d0*dlog10(c6) + dlog10(Pg) - 0.7*dlog10(T))
		else
			gamma6 = 0.d0
		endif

		cte = 1.d-8 / (4.d0 * PI * PC)
		damping_out = cte * (gamma6+gamma_nat+gamma_s) * lambda**2 / (doppler_width * lambda / PC)

		deallocate(gamma6)
		deallocate(gamma_nat)
		deallocate(gamma_s)		

	end subroutine calculate_damping

!-----------------------------------------------------------------
! Fill the strength and splitting variables of the atomic_transition structure
!-----------------------------------------------------------------
	subroutine generate_atomic_zeeman_components(at)
	type(atomic_transition_type) :: at
	integer :: i, n, nlow, nup, iup, ilow, i_pi, i_red, i_blue, cual
	real(kind=8) :: Mup, Mlow, strength


		nup = 2*at%J_up+1
		nlow = 2*at%J_low+1

		i_pi = 0
		i_blue = 0
		i_red = 0

! First count the number of components of each type (
		at%n_components = 0
		do iup = 1, nup
			Mup = at%J_up + 1 - iup  ! Mup=J...-J
			do ilow = 1, 3
				Mlow = Mup-2+ilow			! Mlow=Mup-1,Mup,Mup+1 (the allowed transitions)
				if (abs(Mlow) <= at%J_low) then
					at%n_components(ilow) = at%n_components(ilow)+1
				endif
			enddo
		enddo

		allocate(at%splitting(3,maxval(at%n_components)))
		allocate(at%strength(3,maxval(at%n_components)))

! Now generate all the data
		do iup = 1, nup
			Mup = at%J_up + 1 - iup  ! Mup=J...-J
			do ilow = 1, 3
				Mlow = Mup-2+ilow			! Mlow=Mup-1,Mup,Mup+1 (the allowed transitions)
				if (abs(Mlow) <= at%J_low) then
					if (ilow == 1) then
						i_blue = i_blue + 1
						cual = i_blue
					endif
					if (ilow == 2) then
						i_pi = i_pi + 1
						cual = i_pi
					endif
					if (ilow == 3) then
						i_red = i_red + 1
						cual = i_red
					endif

					at%strength(ilow,cual) = strength_zeeman(at%J_up,at%J_low,Mup,Mlow)
					at%splitting(ilow,cual) = (at%lande_up*Mup - at%lande_low*Mlow)

				endif
			enddo
		enddo

	end subroutine generate_atomic_zeeman_components

	
!-----------------------------------------------------------------
! Select the lines between wmin and wmax in A filling the global arrays
!
!
!FORMAT(F11.4,F7.3,F6.2,F12.3,F5.2,1X,A10,F12.3,F5.2,1X,A10,
!3F6.2,A4,2I2,I3,F6.3,I3,F6.3,2I5,1X,A1,A1,1X,A1,A1,i1,A3.2I5,I6)
! 1 wavelength (nm)  air above 200 nm   F11.4
! 2 log gf  F7.3
! 3 element code = element number + charge/100.  F6.2
! 4 first energy level in cm-1	F12.3
! 5 J for first level	F5.1
!   blank for legibility	1X
! 6 label field for first level   A10
! 7 second energy level in cm-1   F12.3
!		  (negative energies are predicted or extrapolated}
! 8 J for second level   F5.1
!   blank for legibility	1X
! 9 label field for second level   A10
!10 log of radiative damping constant, Gamma Rad  F6.2 or F6.3
!11 log of stark damping constant/electron number. Gamma Stark  F6.2 or F6.3
!12 log of van der Waals damping constant/neutral hydrogen number,
!  	  Gamma van der Waals	F6.2 or F6.3
!13 reference that can be expanded in subdirectory LINES   A4
!14 non-LTE level index for first level	I2
!15 non-LTE level index for second level   I2
!16 isotope number	I3
!17 hyperfine component log fractional strength  F6.3
!18 isotope number  (for diatomics there are two and no hyperfine)	I3
!19 log isotopic abundance fraction   F6.3
!20 hyperfine shift for first level in mK to be added to E  I5
!21 hyperfine shift for second level in mK to be added to E'  I5
!   the symbol "F" for legibilty   1X
!22 hyperfine F for the first level 	I1
!23 note on character of hyperfine data for first level: z none, ? guessed  A1
!   the symbol "-" for legibility	 1X
!24 hyperfine F' for the second level  I1
!25 note on character of hyperfine data for second level: z none, ? guessed  A1
!26 1-digit code, sometimes for line strength classes   I1
!27 3-character code such as AUT for autoionizing    A3
!28 lande g for first level times 1000   I5
!29 lande g for second level times 1000	I5
!30 isotope shift of wavelength in mA
! WARNING!!!
!  In this linelist, the first level is not always the level with the lowest energy
!  Therefore, we have to be sure that we take the excitation potential of the lower level
!  and that the Lande factors and J of the levels are associated correctly
!  I SHOULD CHANGE THE NAMES OF THE VARIABLES TO _first and _second instead of _up and _low
!-----------------------------------------------------------------
	subroutine read_lines(atm, spectrum)	
	type(atmosphere_type) :: atm
	type(spectrum_type) :: spectrum
	real(kind=8) :: wlength, lgf, elem, ioniz, isot_ab
	real(kind=8) :: j_low, e_low, e_up, j_up, gamma_rad, gamm_stark, vdw_damp, hyperf_str, dnumax
	character(len=10) :: label_low, label_up
	character(len=4) :: reference
	character(len=1) :: hpflow_c, hpfup_c
	character(len=3) :: autoion
	integer :: nonlte_up, nonlte_low, isot, isot2, hpf_low, hpf_up, hpff_low, hpff_up, line_str
	integer :: i, j, lande_up, lande_low, isot_shift, lower_level
	integer :: index_from, index_to, index_from_old, index_to_old, indexx, el_dip, from_to(2)
	logical :: found_from, found_to, incluir


! Read the index to recognize the place to jump from the whole list
		open(unit=10,file=trim(adjustl(conf%linelist_file))//'.index',status='old',action='read')
		found_from = .false.
		found_to = .false.
		indexx = 0
		
		do while(.not.found_from .or. .not.found_to)
			if (.not.found_from) index_from = indexx
			if (.not.found_to) index_to = indexx
			read(10,*) wlength, indexx
			if (wlength >= spectrum%ini_lambda .and. .not.found_from) then
				found_from = .true.
			endif
			if (wlength >= spectrum%end_lambda .and. .not.found_to) then
				found_to = .true.
				index_to = indexx
			endif
		enddo
		close(10)
		
		open(unit=10,file=trim(adjustl(conf%linelist_file)),status='old',action='read')

! Jump to the place indicated by the index
		do i = 1, index_from
			read(10,*)
		enddo
		
		j = 0
		do i = 1, index_to-index_from
			read(10,FMT='(F11.4,F7.3,F6.2,F12.3,F6.1,1X,A10,F12.3,F6.1,1X,A10,3F6.2,A4,2I2,I3,F6.3,I3,&
				&F6.3,2I5,1X,A1,A1,1X,A1,A1,i1,A3,2I5,I6)') wlength, lgf, elem, e_low, j_low, label_low,&
				e_up, j_up, label_up, gamma_rad, gamm_stark, vdw_damp, reference, nonlte_up,&
				nonlte_low, isot, hyperf_str, isot2, isot_ab, hpf_low, hpf_up, hpff_low,&
				hpflow_c, hpff_up, hpfup_c, line_str, autoion, lande_up, lande_low, isot_shift
			wlength = wlength * 10.d0
			
! Test for molecules
! Molecules are indicated with an element < 1
! 0.01 for CO
! 0.02 for OH
! 0.03 for CN
! 0.04 for TiO
			if (elem < 1) then
				ioniz = 1
			else
				ioniz = floor((elem - int(elem)) * 100.d0+0.01d0)
			endif

! Take into account 1 A above the limits to include lines outside the region
! we want to synthesize
			incluir = .FALSE.
			if (wlength < spectrum%end_lambda+1 .and. wlength > spectrum%ini_lambda-1 .and. ioniz <= 1) then
				incluir = .TRUE.
			endif

			if (incluir) then
				j = j + 1
			endif
		enddo
		
 		spectrum%nlines = j

 		rewind(10)
 
! Jump to the place indicated by the index
		do i = 1, index_from
			read(10,*)
		enddo

		allocate(spectrum%line(spectrum%nlines))

		j = 0
		el_dip = 0
		do i = 1, index_to-index_from !564949-index_from
			read(10,FMT='(F11.4,F7.3,F6.2,F12.3,F6.1,1X,A10,F12.3,F6.1,1X,A10,3F6.2,A4,2I2,I3,F6.3,I3,&
				&F6.3,2I5,1X,A1,A1,1X,A1,A1,i1,A3,2I5,I6)') wlength, lgf, elem, e_low, j_low, label_low,&
				e_up, j_up, label_up, gamma_rad, gamm_stark, vdw_damp, reference, nonlte_up,&
				nonlte_low, isot, hyperf_str, isot2, isot_ab, hpf_low, hpf_up, hpff_low,&
				hpflow_c, hpff_up, hpfup_c, line_str, autoion, lande_low, lande_up, isot_shift

! This trick solves some problems with the rounding
! Test for molecules
			if (elem < 1) then
				ioniz = 1
			else
				ioniz = floor((elem - int(elem)) * 100.d0+0.01d0)
			endif
			
			wlength = wlength * 10.d0

			incluir = .FALSE.
   
			if (wlength < spectrum%end_lambda+1 .and. wlength > spectrum%ini_lambda-1 .and. ioniz <= 1) then
				incluir = .TRUE.
			endif

			if (incluir) then
				j = j + 1
				lower_level = 1
				if (e_up < e_low) lower_level = 2
				spectrum%line(j)%wlength = wlength
				spectrum%line(j)%freq = PC / (wlength * 1.d-8)
				spectrum%line(j)%gf = 10.d0**lgf
				
				if (lower_level == 1) then    ! The lower level is the first one
					spectrum%line(j)%expot = dabs(e_low) * PH * PC
					spectrum%line(j)%lande_up = lande_up / 1000.d0
					spectrum%line(j)%lande_low = lande_low / 1000.d0
					spectrum%line(j)%J_up = j_up
					spectrum%line(j)%J_low = j_low
				else  								! The lower level is the second one
					spectrum%line(j)%expot = dabs(e_up) * PH * PC
					spectrum%line(j)%lande_up = lande_low / 1000.d0
					spectrum%line(j)%lande_low = lande_up / 1000.d0
					spectrum%line(j)%J_up = j_low
					spectrum%line(j)%J_low = j_up
				endif
				
! Molecules are elements above 100
! CO is 101
! OH is 102
! CN is 103
! TiO is 104
				if (elem < 1) then
					spectrum%line(j)%element = int(100*elem)+100
! CO
					if (abs(elem-0.01) < 1.d-4) then
						spectrum%line(j)%mass = atm%mass(6)+atm%mass(8)   ! CO
						spectrum%line(j)%molecule = 1
					endif

! OH
					if (abs(elem-0.02) < 1.d-4) then
						spectrum%line(j)%mass = atm%mass(6)+atm%mass(1)   ! OH
						spectrum%line(j)%molecule = 2
					endif

! CN
					if (abs(elem-0.03) < 1.d-4) then
						spectrum%line(j)%mass = atm%mass(6)+atm%mass(1)   ! CN
						spectrum%line(j)%molecule = 3
					endif
					
! TiO
					if (abs(elem-0.04) < 1.d-4) then
						spectrum%line(j)%mass = atm%mass(46)+atm%mass(8)   ! TiO
						spectrum%line(j)%molecule = 4
					endif
					
					spectrum%line(j)%atomic_molecular = 2
				else
					spectrum%line(j)%element = int(elem)
					spectrum%line(j)%mass = atm%mass(int(elem))
					spectrum%line(j)%atomic_molecular = 1

				endif

				spectrum%line(j)%ionization = ioniz

! Put these values in case we will use them
				spectrum%line(j)%gamma_rad = 10.d0**gamma_rad
				spectrum%line(j)%gamma_stark = 10.d0**gamm_stark
				spectrum%line(j)%gamma_vdw = 10.d0**vdw_damp

				spectrum%line(j)%enhacement_vdw = 1.d0


! Find the limits of the atomic line
				dnumax = maxval(spectrum%line(j)%freq * dsqrt(atm%microturb**2.d0 + 2.d0 * PK * atm%T / (spectrum%line(j)%mass * UMA)) / PC)

				from_to = extremes(spectrum%freq, spectrum%line(j)%freq, 10.d0*dnumax)

				spectrum%line(j)%lambda_from = from_to(1)
				spectrum%line(j)%lambda_to = from_to(2)

! Compute the Zeeman components if a magnetic synthesis is done
				if (index(atm%magnetic, 'NONMAGNETIC') == 0) then
					call generate_atomic_zeeman_components(spectrum%line(j))
				endif

			endif
		enddo
		
		close(10)

	end subroutine read_lines

!-----------------------------------------------------------------
! Return the profiles weighted by the strength of the components for a given frequency
! It returns zeeman_profile(q,n_depths) with
!  q=1  Mlow=Mup-1  (sigma blue)
!  q=2  Mlow=Mup    (sigma pi)
!  q=3  Mlow=Mup+1  (sigma red)
!-----------------------------------------------------------------
	subroutine zeeman_profile(damping,freq,at,vmacros,B_field,dnu,zeeman_voigt,zeeman_faraday)
	real(kind=8) :: freq
	type(atomic_transition_type) :: at
	real(kind=8) :: damping(:), vmacros(:), dnu(:), B_field(:)
	real(kind=8) :: zeeman_voigt(3,size(damping)), zeeman_faraday(3,size(damping))
	integer :: n, nlow, nup, iup, ilow, i_pi, i_blue, i_red, cual
	real(kind=8) :: Mup, Mlow, strength
	real(kind=8), allocatable :: splitting(:), profile(:,:)

		n = size(damping)

		allocate(splitting(n))
		allocate(profile(2,n))

		nup = 2*at%J_up+1
		nlow = 2*at%J_low+1

		zeeman_voigt = 0.d0
		zeeman_faraday = 0.d0

		i_red = 0
		i_pi = 0
		i_blue = 0

		do iup = 1, nup
			Mup = at%J_up + 1 - iup  ! Mup=J...-J
			do ilow = 1, 3
				Mlow = Mup-2+ilow			! Mlow=Mup-1,Mup,Mup+1 (the allowed transitions)
				if (abs(Mlow) <= at%J_low) then

					if (ilow == 1) then
						i_blue = i_blue + 1
						cual = i_blue
					endif
					if (ilow == 2) then
						i_pi = i_pi + 1
						cual = i_pi
					endif
					if (ilow == 3) then
						i_red = i_red + 1
						cual = i_red
					endif

					strength = at%strength(ilow,cual)

					splitting = LARMOR * B_field * at%splitting(ilow,cual)

					profile = vecfvoigt_zeeman(damping,(at%freq - freq - at%freq*vmacros / PC + splitting ) / dnu)

					zeeman_voigt(ilow,:) = zeeman_voigt(ilow,:) + strength * profile(1,:)
					zeeman_faraday(ilow,:) = zeeman_faraday(ilow,:) + strength * profile(2,:)
				endif
			enddo
		enddo

		deallocate(splitting)
		deallocate(profile)

	end subroutine zeeman_profile

!-----------------------------------------------------------------
! Return the seven independent elements of the absorption matrix
! Remember that, zeeman_voigt(q,:) and zeeman_faraday(q,:) have
!  q=1  Mlow=Mup-1  (sigma blue)
!  q=2  Mlow=Mup    (sigma pi)
!  q=3  Mlow=Mup+1  (sigma red)
!-----------------------------------------------------------------
	subroutine zeeman_opacity(zeeman_voigt,zeeman_faraday,atmos,coefficients)
	real(kind=8) :: zeeman_voigt(:,:), zeeman_faraday(:,:), coefficients(:,:)
	type(atmosphere_type) :: atmos

! Classical absorption coefficients
		coefficients(1,:) = 0.5d0 * (zeeman_voigt(2,:)*atmos%sin_gamma**2 + &
			0.5d0*(zeeman_voigt(1,:)+zeeman_voigt(3,:))*(1.d0+atmos%cos_gamma**2))  ! eta_I
		coefficients(2,:) = 0.5d0 * (zeeman_voigt(2,:) - &
			0.5d0*(zeeman_voigt(1,:)+zeeman_voigt(3,:))) * atmos%sin_gamma**2*atmos%cos_2chi  ! eta_Q
		coefficients(3,:) = 0.5d0 * (zeeman_voigt(2,:) - &
			0.5d0*(zeeman_voigt(1,:)+zeeman_voigt(3,:))) * atmos%sin_gamma**2*atmos%sin_2chi  ! eta_U
		coefficients(4,:) = 0.5d0 * (zeeman_voigt(3,:)-zeeman_voigt(1,:)) * atmos%cos_gamma  ! eta_V

! Magneto-optical coefficients
		coefficients(5,:) = 0.5d0 * (zeeman_faraday(2,:) - &
			0.5d0*(zeeman_faraday(1,:)+zeeman_faraday(3,:))) * atmos%sin_gamma**2*atmos%cos_2chi  ! rho_Q
		coefficients(6,:) = 0.5d0 * (zeeman_faraday(2,:) - &
			0.5d0*(zeeman_faraday(1,:)+zeeman_faraday(3,:))) * atmos%sin_gamma**2*atmos%sin_2chi  ! rho_U
		coefficients(7,:) = 0.5d0 * (zeeman_faraday(3,:)-zeeman_faraday(1,:)) * atmos%cos_gamma  ! rho_V

	end subroutine zeeman_opacity


!-----------------------------------------------------------------
! Estimate the integration ranges of each line
!-----------------------------------------------------------------
	function estimate_range_line(lambda, dnu, damping, at_opacity, height)
	real(kind=8) :: lambda, dnu(:), damping(:), at_opacity(:), height(:), doppler_widths, estimate_range_line, tau, chiO, chiM
	logical :: done
	integer :: loop, n, i

		n = size(height)
		
! Go over a set of widhts and test whether the total opacity of the line is small enough
! so that we have arrived at the continuum
		doppler_widths = 1.d0
		done = .FALSE.
		do while (.not. done)			
			tau = 0.d0

			chiM = at_opacity(1) * fvoigt(damping(1), doppler_widths) / (dnu(1) * SQRTPI)
			
			do i = 2, n				
				chiO = at_opacity(i) * fvoigt(damping(i), doppler_widths) / (dnu(i) * SQRTPI)
				
				tau = tau + abs(height(i)-height(i-1)) * 0.5d0 * (chiO + chiM)

				chiM = chiO
			enddo

! 			print *, lambda, doppler_widths, tau, maxval(dnu)

			if (tau < 1.d-1) then
				done = .TRUE.
			endif

			doppler_widths = doppler_widths + 1.d0
		enddo

		estimate_range_line = doppler_widths
	
	end function estimate_range_line

!-----------------------------------------------------------------
! Add line opacity to the total opacity
!-----------------------------------------------------------------
	subroutine add_line_opacity(atm, spectrum, mu)
	type(config_type) :: conf
	type(atmosphere_type) :: atm
	type(spectrum_type) :: spectrum
	real(kind=8) :: mu, dnumax

	real(kind=8), allocatable :: u1(:), u2(:), u3(:), n1overn0(:), n2overn1(:), niovern(:), at_opacity(:)
	real(kind=8), allocatable :: dnu(:), doppler_atomic(:), perfil(:), ui(:), damping(:)
	integer, allocatable :: from_to(:,:)
	real(kind=8), allocatable :: zeeman_voigt(:,:), zeeman_faraday(:,:), coefficients(:,:)
	integer :: i, j, k, from_to_line(2)


		allocate(u1(atm%n_depths))
		allocate(u2(atm%n_depths))
		allocate(u3(atm%n_depths))
		allocate(ui(atm%n_depths))
		allocate(n1overn0(atm%n_depths))
		allocate(n2overn1(atm%n_depths))
		allocate(niovern(atm%n_depths))
		allocate(at_opacity(atm%n_depths))
		allocate(damping(atm%n_depths))
		allocate(dnu(atm%n_depths))
		allocate(doppler_atomic(atm%n_depths))
		allocate(perfil(atm%n_depths))

! If magnetic, allocate memory for the Voigt and Faraday profiles
		if (index(atm%magnetic, 'NONMAGNETIC') == 0) then
			allocate(zeeman_voigt(3,atm%n_depths))
			allocate(zeeman_faraday(3,atm%n_depths))
			allocate(coefficients(7,atm%n_depths))		
			zeeman_voigt = 0.d0
			zeeman_faraday = 0.d0

! Additionally, compute the angles between the magnetic field and the LOS
! for the inclination angle of the LOS and the angles of B wrt to the local vertical
			atm%cos_gamma = (1.d0-mu**2) * sin(atm%thetaB) * cos(atm%chiB) + mu * cos(atm%thetaB)
			atm%sin_gamma = sqrt(1.d0 - atm%cos_gamma**2)
			atm%cos_2chi = cos(2.d0*atm%chiB)
			atm%sin_2chi = sin(2.d0*atm%chiB)
		endif

! Calculate the line opacity
		do i = 1, spectrum%nlines

! Compute line integrated opacity
! If atomic line			
			if (spectrum%line(i)%atomic_molecular == 1) then
				u1 = atm%partition_functions(:, spectrum%line(i)%element, 1)
				u2 = atm%partition_functions(:, spectrum%line(i)%element, 2)
				u3 = atm%partition_functions(:, spectrum%line(i)%element, 3)

				n1overn0 = saha(atm%T, PK * atm%ne * atm%T, u1, u2, atm%ionization_pot(1,spectrum%line(i)%element))
				n2overn1 = saha(atm%T, PK * atm%ne * atm%T, u2, u3, atm%ionization_pot(2,spectrum%line(i)%element))

				if (spectrum%line(i)%ionization == 0) then
					niovern = 1.d0 / (1.d0 + n1overn0 + n2overn1 * n1overn0)
					ui = u1
				else if (spectrum%line(i)%ionization == 1) then
					niovern = 1.d0 / (1.d0 + 1.d0/n1overn0 + n2overn1)
					ui = u2
				else
					print *, 'Unknown ionization stage...'
					niovern = 0.d0
				endif

! Compute the line center opacity
				at_opacity = OPA * spectrum%line(i)%gf / ui * dexp(-spectrum%line(i)%expot / (PK * atm%T)) *&
									(1.d0 - dexp(-PHK * spectrum%line(i)%freq / atm%T)) * niovern * &
									(atm%nhtot * atm%abundances(spectrum%line(i)%element))

! Doppler width in velocity and frequency units
				doppler_atomic = dsqrt(atm%microturb**2.d0 + 2.d0 * PK * atm%T / (spectrum%line(i)%mass * UMA))
				dnu = spectrum%line(i)%freq * doppler_atomic / PC

! Compute damping
				call calculate_damping(atm%T, atm%nHtot, atm%Pe, atm%P_total, doppler_atomic, &
					atm%ionization_pot(spectrum%line(i)%ionization+1,spectrum%line(i)%element), spectrum%line(i)%expot/EV_ERG, &
					spectrum%line(i)%gamma_rad, spectrum%line(i)%gamma_stark, spectrum%line(i)%gamma_vdw, spectrum%line(i)%wlength, damping)

! Try to estimate the range of wavelengths in which the line has opacity contribution
				dnumax = estimate_range_line(spectrum%line(i)%wlength, dnu, damping, at_opacity, atm%height) * maxval(dnu)

! If the line is estimated to be very broad, take into account that the result
! might be truncated if the incorrect chunks are used
				from_to_line = extremes(spectrum%freq, spectrum%line(i)%freq, dnumax)

				spectrum%line(i)%lambda_from = from_to_line(1)
				spectrum%line(i)%lambda_to = from_to_line(2)
					
			else

! If molecular line				
				at_opacity = OPA * spectrum%line(i)%gf / atm%partition_functions_molecular(:,spectrum%line(i)%molecule) * &
									dexp(-spectrum%line(i)%expot / (PK * atm%T)) *&
									(1.d0 - dexp(-PHK * spectrum%line(i)%freq / atm%T)) * &
									atm%mol_density(spectrum%line(i)%molecule,:)
																		
				doppler_atomic = dsqrt(atm%microturb**2.d0 + 2.d0 * PK * atm%T / (spectrum%line(i)%mass * UMA))
				dnu = spectrum%line(i)%freq * doppler_atomic / PC

				damping = 0.d0
									
			endif
 


! Compute opacity
			if (index(atm%magnetic, 'NONMAGNETIC') /= 0) then
			
! Non-magnetic			
				do k = spectrum%line(i)%lambda_from, spectrum%line(i)%lambda_to
					perfil = vecfvoigt(damping,(spectrum%line(i)%freq - spectrum%freq(k) - spectrum%line(i)%freq * atm%vmacro / PC) / dnu)
					spectrum%chi(1,:,k) = spectrum%chi(1,:,k) + at_opacity * perfil / (dnu * SQRTPI)
				enddo
				
			else
			
! Magnetic		
! Do not do the full calculation for a non-magnetic line
				if (spectrum%line(i)%lande_up == 0.d0 .and. spectrum%line(i)%lande_low == 0.d0) then
					do k = spectrum%line(i)%lambda_from, spectrum%line(i)%lambda_to
						perfil = vecfvoigt(damping,(spectrum%line(i)%freq - spectrum%freq(k) - spectrum%line(i)%freq * atm%vmacro / PC) / dnu)
						spectrum%chi(1,:,k) = spectrum%chi(1,:,k) + at_opacity * perfil / (dnu * SQRTPI)						
					enddo
				else
					do k = spectrum%line(i)%lambda_from, spectrum%line(i)%lambda_to
						call zeeman_profile(damping, spectrum%freq(k), spectrum%line(i), atm%vmacro, atm%B, dnu, zeeman_voigt, zeeman_faraday)
						call zeeman_opacity(zeeman_voigt, zeeman_faraday, atm, coefficients)
						
						do j = 1, 7
							spectrum%chi(j,:,k) = spectrum%chi(j,:,k) + at_opacity * coefficients(j,:) / (dnu * SQRTPI)
						enddo
						
					enddo
				endif
				
			endif

		enddo

		deallocate(u1)
		deallocate(u2)
		deallocate(u3)
		deallocate(ui)
		deallocate(n1overn0)
		deallocate(n2overn1)
		deallocate(niovern)
		deallocate(at_opacity)
		deallocate(dnu)		
		deallocate(doppler_atomic)
		deallocate(perfil)
		deallocate(damping)

		if (index(atm%magnetic, 'NONMAGNETIC') == 0) then
			deallocate(zeeman_voigt)
			deallocate(zeeman_faraday)
			deallocate(coefficients)
		endif
		
	end subroutine add_line_opacity

!-----------------------------------------------------------------
! Add line opacity to the total opacity
!-----------------------------------------------------------------
	subroutine add_background_opacity(atm, spectrum)
	integer :: i, j
	type(atmosphere_type) :: atm
	type(spectrum_type) :: spectrum

		spectrum%chi = 0.d0
		
		do i = 1, atm%n_depths
			do j = 1, spectrum%nlambda				
				spectrum%chi(1,i,j) = background_opacity(atm%T(i), atm%Pe(i), atm%PH(i), atm%PHminus(i), &
					atm%PHplus(i), atm%PH2(i), atm%PH2plus(i), PC / spectrum%freq(j)*1.d8)
			enddo
		enddo

	end subroutine add_background_opacity

!-----------------------------------------------------------------
! Add line opacity to the total opacity
!-----------------------------------------------------------------
	subroutine solve_RT(atm, spectrum, mu, weight, which_mu, spec_cont)
	integer :: i, which_mu, spec_cont
	type(atmosphere_type) :: atm
	type(spectrum_type) :: spectrum
	real(kind=8) :: mu, weight

		spectrum%boundary = 0.d0

! Source function and boundary condition		
		do i = 1, spectrum%nlambda
			spectrum%source(:,i) = planck_vect_lambda(PC/spectrum%freq(i),atm%T)
			spectrum%boundary(1,i) = planck_lambda(PC / spectrum%freq(i), atm%T(atm%n_depths))
		enddo
		
! Non-magnetic			
		if (index(atm%magnetic, 'NONMAGNETIC') /= 0) then
			if (index(conf%mode, 'INTENSITY') /= 0) then
				spectrum%flux(spec_cont,which_mu,:) = weight * formal_sol_vect(atm%height, spectrum%chi(1,:,:), spectrum%source(:,:), mu, spectrum%boundary(1,:), atm%direction)
			else
				spectrum%flux(spec_cont,which_mu,:) = spectrum%flux(spec_cont,which_mu,:) + &
					weight * formal_sol_vect(atm%height, spectrum%chi(1,:,:), spectrum%source(:,:), mu, spectrum%boundary(1,:), atm%direction)
			endif
			

		else
! Magnetic
			if (index(conf%mode, 'INTENSITY') /= 0) then
				do i = 1, spectrum%nlambda
					spectrum%flux(:,which_mu,i) = weight * formal_sol_polarized(atm%height, spectrum%chi(:,:,i), spectrum%source(:,i), mu, spectrum%boundary(:,i), atm%direction)	
				enddo
			else
				do i = 1, spectrum%nlambda
					spectrum%flux(:,which_mu,i) = spectrum%flux(:,which_mu,i) + &
						weight * formal_sol_polarized(atm%height, spectrum%chi(:,:,i), spectrum%source(:,i), mu, spectrum%boundary(:,i), atm%direction)	
				enddo
			endif

		endif
						
	end subroutine solve_RT

!-----------------------------------------------------------------
! Generate wavelength axis
!-----------------------------------------------------------------
	subroutine setup_spectrum(conf, atm, spectrum)
	type(atmosphere_type) :: atm
	type(spectrum_type) :: spectrum
	type(config_type) :: conf
	integer :: i

		spectrum%nlambda = (spectrum%end_lambda - spectrum%ini_lambda) / conf%step_lambda
		allocate(spectrum%source(atm%n_depths, spectrum%nlambda))

		allocate(spectrum%freq(spectrum%nlambda))

		if (index(atm%magnetic, 'NONMAGNETIC') /= 0) then
! Non-magnetic			
			allocate(spectrum%flux(2,conf%nmus_intensity,spectrum%nlambda))
			allocate(spectrum%chi(1,atm%n_depths, spectrum%nlambda))
			allocate(spectrum%boundary(1,spectrum%nlambda))

		else
! Magnetic
			allocate(spectrum%flux(4,conf%nmus_intensity,spectrum%nlambda))
			allocate(spectrum%chi(7,atm%n_depths, spectrum%nlambda))
			allocate(spectrum%boundary(4,spectrum%nlambda))

		endif

! Generate the frequency axis
 		do i = 1, spectrum%nlambda
 			spectrum%freq(i) = PC / (1.d-8 * (spectrum%ini_lambda + (i-1.d0) * conf%step_lambda))
 		enddo

 	end subroutine setup_spectrum

end module atomic_mod