module io_mod
use vars_mod, only : config_file, myrank, PK, UMA, identity, PI, nMolecules, mol_code
use types_mod, only : config_type, atmosphere_type
use chemical_eq_mod, only : calculate_abundance_pg_from_t_pe
use atomic_partition_mod, only : partition_atomic
use maths_mod, only : spline
implicit none
contains

!---------------------------------------------------------
! Read the input file and set up the variables
!---------------------------------------------------------
	subroutine read_config_file(conf)
	type(config_type) :: conf
	integer :: nargs, j

! Verify if a configuration file is passed as an argument
! If not, use the default 'config' file
		nargs = iargc()

		if (nargs == 0) then	      
			config_file = 'conf.input'
		endif

		if (nargs == 1) then	      
			call getarg(1,config_file)			
		endif

! Read the configuration file
		open(unit=12,file=config_file,action='read',status='old')
		read(12,*) conf%mode		
		read(12,*) conf%nmus
				
		allocate(conf%mus(conf%nmus))
		allocate(conf%weights(conf%nmus))

		read(12,*) (conf%mus(j),j=1,conf%nmus)
		read(12,*) (conf%weights(j),j=1,conf%nmus)
				
		read(12,*) conf%atm_file
		read(12,*) conf%linelist_file
		read(12,*) conf%out_file
		
		read(12,*) conf%ini_lambda
		read(12,*) conf%end_lambda
		read(12,*) conf%step_lambda
		read(12,*) conf%chunk_size

		if (index(conf%mode, 'INTENSITY') /= 0) then
			conf%nmus_intensity = conf%nmus
		else
			conf%nmus_intensity = 1
		endif

		close(12)
		
	end subroutine read_config_file

!---------------------------------------------------------
! Read atomic abundances from the common file
!---------------------------------------------------------
	subroutine compute_partition_molecular(atm)
	type(atmosphere_type) :: atm
	real(kind=8), allocatable :: temp(:), part(:)
	integer :: n, i, j

		open(unit=12,file='DATA/partition_molecules.dat',action='read',status='old')
		do j = 1, nMolecules
			read(12,*)
			read(12,*)
			read(12,*) n
			allocate(temp(n))
			allocate(part(n))
			do i = 1, n
				read(12,*) temp(i), part(i)
			enddo
			call spline(temp,part,atm%T,atm%partition_functions_molecular(:,j))

			deallocate(temp, part)
		enddo
		close(12)


	end subroutine compute_partition_molecular
!---------------------------------------------------------
! Read atomic abundances from the common file
!---------------------------------------------------------
	subroutine read_abundance(atm)
	type(atmosphere_type) :: atm
	integer :: i
	real(kind=8) :: j

		open(unit=12,file='DATA/abundances.dat',action='read',status='old')

		allocate(atm%abundances(92))
		allocate(atm%mass(92))
		allocate(atm%ionization_pot(2,92))		
		allocate(atm%element_symbol(92))
		
		read(12,*)

		do i = 1, 92
			read(12,*) atm%element_symbol(i), j, atm%abundances(i), atm%mass(i), atm%ionization_pot(1,i), atm%ionization_pot(2,i)
		enddo
		
		close(12)
	end subroutine read_abundance

!---------------------------------------------------------
! Read the atmosphere
!---------------------------------------------------------
	subroutine read_atmosphere(conf,atm)
	type(config_type) :: conf
	type(atmosphere_type) :: atm
	real(kind=8) :: abund_change(92), abundance_scale, logabundH
	real(kind=8), allocatable, dimension(:) :: u1, u2, u3, rho
	character(len=20) :: abundances_label
	integer :: i, j

		open(unit=12,file=conf%atm_file,action='read',status='old')

		read(12,*)

		read(12,*) atm%hscale

		read(12,*) atm%magnetic

		if (myrank == 0) then
			if (index(atm%magnetic, 'NONMAGNETIC') /= 0) then
				write(*,FMT='(A,A)') 'Reading non-magnetic atmosphere : ', trim(adjustl(conf%atm_file))
			else
				write(*,FMT='(A,A)') 'Reading magnetic atmosphere : ', trim(adjustl(conf%atm_file))
			endif
		endif

! Read modifications to the abundances in case needed
		read(12,*) abundances_label

		if (index(abundances_label,'MODIFIED') /= 0) then

			read(12,*)
			read(12,*) abundance_scale

			read(12,*)
			read(12,*) abund_change

			logabundH = log10(abund_change(1))

			atm%abundances(1) = 12.d0
			atm%abundances(2) = atm%abundances(1) + log10(abund_change(2)) - logabundH

			do i = 3, 92
				atm%abundances(i) = atm%abundances(1) + abund_change(i) - logabundH
			enddo
			
		endif

		atm%abundances = 10.d0**(atm%abundances-12.d0)

! Mean molecular weight
		atm%avgWeight = sum(atm%abundances * atm%mass)

! Go on reading the model atmosphere
		read(12,*) atm%n_depths

		allocate(atm%height(atm%n_depths))
		allocate(atm%cmass(atm%n_depths))
		allocate(atm%T(atm%n_depths))
		allocate(atm%ne(atm%n_depths))
		allocate(atm%Pe(atm%n_depths))
		allocate(atm%microturb(atm%n_depths))
		allocate(atm%vmacro(atm%n_depths))


		allocate(atm%PH(atm%n_depths))
		allocate(atm%PHminus(atm%n_depths))
		allocate(atm%PHplus(atm%n_depths))
		allocate(atm%PH2(atm%n_depths))
		allocate(atm%PH2plus(atm%n_depths))
		allocate(atm%P_total(atm%n_depths))
		allocate(atm%nhtot(atm%n_depths))

		allocate(atm%mol_density(nMolecules,atm%n_depths))

! Column mass scale
		if (index(atm%hscale,'CMASS') /= 0) then

! Magnetic or non-magnetic
			if (index(atm%magnetic, 'NONMAGNETIC') /= 0) then
				do i = 1, atm%n_depths
					read(12,*) atm%cmass(i), atm%T(i), atm%ne(i), atm%microturb(i), atm%vmacro(i)
					atm%Pe(i) = PK * atm%T(i) * atm%ne(i)
				enddo
			else

				allocate(atm%B(atm%n_depths))
				allocate(atm%thetaB(atm%n_depths))
				allocate(atm%chiB(atm%n_depths))
				
				allocate(atm%cos_gamma(atm%n_depths))
				allocate(atm%sin_gamma(atm%n_depths))
				allocate(atm%cos_2chi(atm%n_depths))
				allocate(atm%sin_2chi(atm%n_depths))
				
				do i = 1, atm%n_depths
					read(12,*) atm%cmass(i), atm%T(i), atm%ne(i), atm%microturb(i), atm%vmacro(i), atm%B(i), atm%thetaB(i), atm%chiB(i)
					atm%Pe(i) = PK * atm%T(i) * atm%ne(i)
				enddo

				atm%thetaB = atm%thetaB * PI / 180.d0
				atm%chiB = atm%chiB * PI / 180.d0
				
			endif

		else
! Magnetic or non-magnetic
			if (index(atm%magnetic, 'NONMAGNETIC') /= 0) then
				do i = 1, atm%n_depths
					read(12,*) atm%height(i), atm%T(i), atm%ne(i), atm%microturb(i), atm%vmacro(i)
					atm%Pe(i) = PK * atm%T(i) * atm%ne(i)
				enddo
				atm%height = atm%height * 1e5
			else

				allocate(atm%B(atm%n_depths))
				allocate(atm%thetaB(atm%n_depths))
				allocate(atm%chiB(atm%n_depths))

				allocate(atm%cos_gamma(atm%n_depths))
				allocate(atm%sin_gamma(atm%n_depths))
				allocate(atm%cos_2chi(atm%n_depths))
				allocate(atm%sin_2chi(atm%n_depths))
				
				do i = 1, atm%n_depths
					read(12,*) atm%height(i), atm%T(i), atm%ne(i), atm%microturb(i), atm%vmacro(i), atm%B(i), atm%thetaB(i), atm%chiB(i)
					atm%Pe(i) = PK * atm%T(i) * atm%ne(i)
				enddo
				atm%height = atm%height * 1e5

				atm%thetaB = atm%thetaB * PI / 180.d0
				atm%chiB = atm%chiB * PI / 180.d0
				
			endif
			
		endif

! Compute chemical equilibrium to get the LTE hydrogen populations		
		atm%mol_density = calculate_abundance_Pg_from_T_Pe(mol_code, atm%n_depths, atm%abundances, atm%height, atm%T, atm%Pe, &
			atm%PH, atm%PHminus, atm%PHplus, atm%PH2, atm%PH2plus, atm%P_total)

! Total hydrogen density
		atm%nhtot = (atm%PH + atm%PHminus + atm%PHplus + atm%PH2 + atm%PH2plus) / (PK * atm%T)

! Generate height axis if mode is given in column mass
		if (index(atm%hscale,'CMASS') /= 0) then

			allocate(rho(atm%n_depths))
			rho = UMA * atm%nhtot * atm%avgWeight
			atm%height(1) = 0.d0
			
			do i = 2, atm%n_depths
				atm%height(i) = atm%height(i-1) - 2.d0 * (atm%cmass(i) - atm%cmass(i-1)) / (rho(i-1) + rho(i))
			enddo

			deallocate(rho)
		endif

! Precompute the partition functions for all temperatures in the model
		allocate(atm%partition_functions(atm%n_depths, 92, 3))
		allocate(atm%partition_functions_molecular(atm%n_depths, nMolecules))

		allocate(u1(atm%n_depths))
		allocate(u2(atm%n_depths))
		allocate(u3(atm%n_depths))
		
		do j = 1, 92
			call partition_atomic(atm%T, j, u1, u2, u3)
			atm%partition_functions(:,j,1) = u1
			atm%partition_functions(:,j,2) = u2
			atm%partition_functions(:,j,3) = u3
		enddo

		call compute_partition_molecular(atm)

		deallocate(u1,u2,u3)

		close(12)

! Try to guess the direction of the atmosphere. We consider the lower boundary
! at the point where the electron density is largest. This should work for
! almost all cases, except in very exotic atmospheres
		if (atm%ne(1) > atm%ne(atm%n_depths)) then
			atm%direction = 1
		else
			atm%direction = -1
		endif
		
	end subroutine read_atmosphere
	
end module io_mod