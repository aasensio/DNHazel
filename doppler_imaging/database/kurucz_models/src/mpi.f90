module mpi_mod
use types_mod, only : config_type, spectrum_type, atmosphere_type
use vars_mod, only : PC, atm, conf, spectrum, nMolecules
implicit none

contains


!------------------------------------------------------------
! Broadcast some general information
!------------------------------------------------------------
	subroutine mpi_bcast_general(myrank)
	integer :: ierr, myrank
	include 'mpif.h'
		
		call MPI_Bcast(conf%step_lambda,1,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,ierr)
		call MPI_Bcast(conf%nmus,1,MPI_INTEGER,0,MPI_COMM_WORLD,ierr)
		call MPI_Bcast(conf%nmus_intensity,1,MPI_INTEGER,0,MPI_COMM_WORLD,ierr)
		call MPI_Bcast(conf%mode,10,MPI_CHARACTER,0,MPI_COMM_WORLD,ierr)
				
		if (myrank /= 0) then
			allocate(conf%mus(conf%nmus))
			allocate(conf%weights(conf%nmus))
		endif

		call MPI_Bcast(conf%mus,conf%nmus,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,ierr)
		call MPI_Bcast(conf%weights,conf%nmus,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,ierr)
		call MPI_Bcast(conf%linelist_file,100,MPI_CHARACTER,0,MPI_COMM_WORLD,ierr)
				
		
 	end subroutine mpi_bcast_general
 	
!------------------------------------------------------------
! Broadcast abundances
!------------------------------------------------------------
	subroutine mpi_bcast_abundances(myrank)
	integer :: ierr, myrank
	include 'mpif.h'

		if (myrank /= 0) then
			allocate(atm%abundances(92))
			allocate(atm%mass(92))
			allocate(atm%ionization_pot(2,92))
			allocate(atm%element_symbol(92))
		endif
		
		call MPI_Bcast(atm%abundances,92,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,ierr)
		call MPI_Bcast(atm%mass,92,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,ierr)
		call MPI_Bcast(atm%ionization_pot,2*92,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,ierr)
		call MPI_Bcast(atm%element_symbol,2*92,MPI_CHARACTER,0,MPI_COMM_WORLD,ierr)
 		call MPI_Barrier(MPI_COMM_WORLD, ierr)

 	end subroutine mpi_bcast_abundances

!------------------------------------------------------------
! Broadcast atmosphere model
!------------------------------------------------------------
	subroutine mpi_bcast_atmosphere(myrank)
	integer :: ierr, myrank
	include 'mpif.h'

		call MPI_Bcast(atm%hscale,15,MPI_CHARACTER,0,MPI_COMM_WORLD,ierr)
		call MPI_Bcast(atm%magnetic,15,MPI_CHARACTER,0,MPI_COMM_WORLD,ierr)
		call MPI_Bcast(atm%n_depths,1,MPI_INTEGER,0,MPI_COMM_WORLD,ierr)
		call MPI_Bcast(atm%direction,1,MPI_INTEGER,0,MPI_COMM_WORLD,ierr)
		
		if (myrank /= 0) then
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

			allocate(atm%mol_density(4,atm%n_depths))

			allocate(atm%partition_functions(atm%n_depths, 92, 3))

			allocate(atm%partition_functions_molecular(atm%n_depths,nMolecules))

! If Zeeman synthesis, allocate memory for vector magnetic field
			if (index(atm%magnetic,'NONMAGNETIC') == 0) then
				allocate(atm%B(atm%n_depths))
				allocate(atm%thetaB(atm%n_depths))
				allocate(atm%chiB(atm%n_depths))

				allocate(atm%cos_gamma(atm%n_depths))
				allocate(atm%sin_gamma(atm%n_depths))
				allocate(atm%cos_2chi(atm%n_depths))
				allocate(atm%sin_2chi(atm%n_depths))
			endif
			
		endif


		call MPI_Bcast(atm%height,atm%n_depths,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,ierr)
		call MPI_Bcast(atm%cmass,atm%n_depths,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,ierr)
		call MPI_Bcast(atm%T,atm%n_depths,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,ierr)
		call MPI_Bcast(atm%ne,atm%n_depths,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,ierr)
		call MPI_Bcast(atm%Pe,atm%n_depths,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,ierr)
		call MPI_Bcast(atm%microturb,atm%n_depths,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,ierr)
		call MPI_Bcast(atm%vmacro,atm%n_depths,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,ierr)
		call MPI_Bcast(atm%PH,atm%n_depths,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,ierr)
		call MPI_Bcast(atm%PHminus,atm%n_depths,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,ierr)
		call MPI_Bcast(atm%PHplus,atm%n_depths,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,ierr)
		call MPI_Bcast(atm%PH2,atm%n_depths,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,ierr)
		call MPI_Bcast(atm%PH2plus,atm%n_depths,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,ierr)
		call MPI_Bcast(atm%P_total,atm%n_depths,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,ierr)
		call MPI_Bcast(atm%nhtot,atm%n_depths,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,ierr)

		call MPI_Bcast(atm%mol_density,4*atm%n_depths,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,ierr)

		call MPI_Bcast(atm%partition_functions,atm%n_depths*92*3,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,ierr)
		call MPI_Bcast(atm%partition_functions_molecular,atm%n_depths*nMolecules,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,ierr)

! If Zeeman synthesis, broadcast vector magnetic field
		if (index(atm%magnetic,'NONMAGNETIC') == 0) then
			call MPI_Bcast(atm%B,atm%n_depths,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,ierr)
			call MPI_Bcast(atm%thetaB,atm%n_depths,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,ierr)
			call MPI_Bcast(atm%chiB,atm%n_depths,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,ierr)
			call MPI_Bcast(atm%cos_gamma,atm%n_depths,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,ierr)
			call MPI_Bcast(atm%sin_gamma,atm%n_depths,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,ierr)
			call MPI_Bcast(atm%cos_2chi,atm%n_depths,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,ierr)
			call MPI_Bcast(atm%sin_2chi,atm%n_depths,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,ierr)
		endif
		
 	end subroutine mpi_bcast_atmosphere

!---------------------------------------------------------
! Divide the wavelength range into equal chunks according to the number of processors
!---------------------------------------------------------
	subroutine divide_work(nprocs, conf, atm, spectrum)
	integer :: nprocs, myrank, i
	type(config_type) :: conf
	type(spectrum_type) :: spectrum
	type(atmosphere_type) :: atm
	real(kind=8) :: wavelength_region

! Divide the wavelength axis into regions of chunk_size length
		conf%nchunks = floor((conf%end_lambda - conf%ini_lambda) / conf%chunk_size)

		if (conf%nchunks == 0) then 
			conf%nchunks = 1
		endif

		allocate(conf%chunks(2,conf%nchunks))

		do i = 1, conf%nchunks
			conf%chunks(1,i) = conf%ini_lambda + (i-1) * conf%chunk_size
			conf%chunks(2,i) = conf%ini_lambda + i * conf%chunk_size
		enddo

! Expand last chunk to cover the remaining piece
		if (conf%chunks(2,conf%nchunks) /= conf%end_lambda) then
			conf%chunks(2,conf%nchunks) = conf%end_lambda
		endif

		do i = 1, conf%nchunks
			write(*,FMT='(A,I5,A,F12.2,A,F12.2,A)') 'Chunk ', i, ' -> [', conf%chunks(1,i), ',', conf%chunks(2,i), ']'
		enddo


		open(unit=24,file=conf%out_file,action='write',status='replace',access='stream')
		write(24) conf%nchunks

	end subroutine divide_work

!------------------------------------------------------------
! Send a new observation to a slave
!------------------------------------------------------------
	subroutine send_new_computation(conf, chunk, slave, packagesize)
	type(config_type) :: conf
	integer :: chunk, slave, packagesize
	integer :: ierr, pos
	character :: buffer(packagesize)
	character(len=8) :: date
	character(len=10) :: time
	character(len=5) :: zone
	integer :: values(8)
	include 'mpif.h'

		pos = 0

		call MPI_Pack(slave, 1, MPI_INTEGER, buffer, packagesize, pos, MPI_COMM_WORLD,ierr)
		call MPI_Pack(conf%chunks(1,chunk), 1, MPI_DOUBLE_PRECISION, buffer, packagesize, pos, MPI_COMM_WORLD,ierr)
		call MPI_Pack(conf%chunks(2,chunk), 1, MPI_DOUBLE_PRECISION, buffer, packagesize, pos, MPI_COMM_WORLD,ierr)

		call MPI_Send(buffer, packagesize, MPI_PACKED, slave, 10, MPI_COMM_WORLD,ierr)

		call date_and_time(date, time, zone, values)		
		write(*,FMT='(A,A,A,I5,A,F12.2,A,F12.2,A)') '(t=',time,') * Master -> Slave ', slave, &
			' - chunk [', conf%chunks(1,chunk), ',', conf%chunks(2,chunk), ']'

 	end subroutine send_new_computation

!------------------------------------------------------------
! Send a new observation to a slave
!------------------------------------------------------------
	subroutine receive_new_computation(initial_lambda, final_lambda, stop_flag, packagesize)
	real(kind=8) :: initial_lambda, final_lambda
	integer :: chunk, slave, stop_flag, packagesize
	integer :: ierr, pos
	character :: buffer(packagesize)
	include 'mpif.h'
	integer :: status(MPI_STATUS_SIZE)
	
		call MPI_Recv(buffer, packagesize, MPI_PACKED, 0, MPI_ANY_TAG, MPI_COMM_WORLD, status, ierr)

		if (status(MPI_TAG) == 10) then
			pos = 0
			call MPI_Unpack(buffer, packagesize, pos, slave, 1, MPI_INTEGER, MPI_COMM_WORLD, ierr)
			call MPI_Unpack(buffer, packagesize, pos, initial_lambda, 1, MPI_DOUBLE_PRECISION, MPI_COMM_WORLD, ierr)
			call MPI_Unpack(buffer, packagesize, pos, final_lambda, 1, MPI_DOUBLE_PRECISION, MPI_COMM_WORLD, ierr)

		endif

! This is the kill flag
		if (status(MPI_TAG) == 999) then
			stop_flag = 1
			return
		endif

 	end subroutine receive_new_computation

!------------------------------------------------------------
! Send the spectrum from a slave to the master
!------------------------------------------------------------
	subroutine send_spectrum(slave)
	integer :: slave
	integer :: ierr, i, nstokes
	character(len=8) :: date
	character(len=10) :: time
	character(len=5) :: zone
	integer :: values(8)
	include 'mpif.h'

		call MPI_Send(slave, 1, MPI_INTEGER, 0, 13, MPI_COMM_WORLD, ierr)
 		call MPI_Send(spectrum%nlambda, 1, MPI_INTEGER, 0, 14, MPI_COMM_WORLD, ierr)
 		call MPI_Send(spectrum%freq, spectrum%nlambda, MPI_DOUBLE_PRECISION, 0, 15, MPI_COMM_WORLD, ierr)

 		if (index(atm%magnetic, 'NONMAGNETIC') /= 0) then			
			nstokes = 2 * conf%nmus_intensity
		else			
			nstokes = 4 * conf%nmus_intensity
		endif
		
 		call MPI_Send(spectrum%flux, nstokes * spectrum%nlambda, MPI_DOUBLE_PRECISION, 0, 16, MPI_COMM_WORLD, ierr)

 		call date_and_time(date, time, zone, values)
		write(*,FMT='(A,A,A,I5,A,F12.2,A,F12.2,A)') '(t=',time,') * Slave ', slave, ' - chunk [', &
			spectrum%ini_lambda, ',', spectrum%end_lambda, '] -- DONE'

 	end subroutine send_spectrum

!------------------------------------------------------------
! Receive the spectrum from a slave
!------------------------------------------------------------
	subroutine receive_spectrum(slave)
	integer :: slave
	integer :: ierr, i, nstokes, j
	include 'mpif.h'
	integer :: status(MPI_STATUS_SIZE)
		
		call MPI_Recv(slave, 1, MPI_INTEGER, MPI_ANY_SOURCE, 13, MPI_COMM_WORLD, status, ierr)
 		call MPI_Recv(spectrum%nlambda, 1, MPI_INTEGER, MPI_ANY_SOURCE, 14, MPI_COMM_WORLD, status, ierr)

 		allocate(spectrum%freq(spectrum%nlambda))

 		if (index(atm%magnetic, 'NONMAGNETIC') /= 0) then
			allocate(spectrum%flux(2,conf%nmus_intensity,spectrum%nlambda))
			nstokes = 2 * conf%nmus_intensity
		else
			allocate(spectrum%flux(4,conf%nmus_intensity,spectrum%nlambda))
			nstokes = 4 * conf%nmus_intensity
		endif
		
 		call MPI_Recv(spectrum%freq, spectrum%nlambda, MPI_DOUBLE_PRECISION, MPI_ANY_SOURCE, 15, MPI_COMM_WORLD, status, ierr)
 		call MPI_Recv(spectrum%flux, nstokes * spectrum%nlambda, MPI_DOUBLE_PRECISION, MPI_ANY_SOURCE, 16, MPI_COMM_WORLD, status, ierr)


 		if (index(atm%magnetic, 'NONMAGNETIC') /= 0) then
			write(24) spectrum%nlambda, conf%nmus_intensity
 			write(24) spectrum%freq, spectrum%flux(1,:,:), spectrum%flux(2,:,:)
		else
			write(24) spectrum%nlambda, conf%nmus_intensity
			write(24) spectrum%freq, spectrum%flux(:,:,:)
		endif

 		deallocate(spectrum%freq)
 		deallocate(spectrum%flux)

 	end subroutine receive_spectrum

!------------------------------------------------------------
! Send the kill signal to all slaves
!------------------------------------------------------------
	subroutine kill_slave(slave)
	integer :: slave
	integer :: ierr, i
	include 'mpif.h'

! Send an empty message with the tag 999
		call MPI_Send(0, 0, MPI_INTEGER, slave, 999, MPI_COMM_WORLD, ierr)

 	end subroutine kill_slave
 	
end module mpi_mod