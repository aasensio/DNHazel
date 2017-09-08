program lte
use io_mod, only : read_config_file, read_atmosphere, read_abundance
use mpi_mod, only : divide_work
use atomic_mod, only : read_lines, add_line_opacity, solve_rt, add_background_opacity, setup_spectrum
use vars_mod, only : myrank, conf, atm, spectrum, PK, identity
use maths_mod, only : factrl
use mpi_mod, only : mpi_bcast_abundances, mpi_bcast_atmosphere, divide_work, send_new_computation, receive_new_computation,&
	mpi_bcast_general, receive_spectrum, kill_slave, send_spectrum
implicit none

	integer :: mpi_status, nprocs, ierr, i, kill_flag, slave, packagesize, j
	character(len=8) :: date
	character(len=10) :: time
	character(len=5) :: zone
	integer :: values(8)
	include 'mpif.h'

	call MPI_INIT(mpi_status)
	call MPI_COMM_RANK(MPI_COMM_WORLD, myrank, mpi_status)
	call MPI_COMM_SIZE(MPI_COMM_WORLD, nprocs, mpi_status)

	write(*,FMT='(A,I3,A,I3)') 'Node ', myrank, '/', nprocs

! Number of processes
	conf%nprocs = nprocs
		
! Status of the slave (active/inactive)
	allocate(conf%slave_killed(conf%nprocs-1))
	conf%slave_killed = 0
		
! Read configuration file. This needs to be done only by the master
	if (myrank == 0) then
		call read_config_file(conf)
	endif

! Read abundances and broadcast to all nodes
	if (myrank == 0) then
 		call read_abundance(atm)
	endif		

! Read atmospheric file
	if (myrank == 0) then
 		call read_atmosphere(conf,atm)
	endif		

! Broadcast information to all nodes	
	call mpi_bcast_general(myrank)	
	call mpi_bcast_abundances(myrank)	
	call mpi_bcast_atmosphere(myrank)	

! Divide work to be done at each node
	if (myrank == 0) then
		call divide_work(nprocs, conf, atm, spectrum)
	endif
! Precompute the factorials
	call factrl
	
! Generate identity matrix
	identity = 0.d0
	do i = 1, 4
		identity(i,i) = 1.d0
	enddo

! Size of package
	packagesize = sizeof(slave) + 2*sizeof(PK)

! Barrier to start computations
	call MPI_Barrier(MPI_COMM_WORLD, ierr)

!*******************************************************
! Start master/slave process if more than two processes
!*******************************************************
	if (nprocs >= 2) then

! Initial distribution of works
 		if (myrank == 0) then

			conf%index_chunk = 1
			
			do slave = 1, min(conf%nchunks, conf%nprocs-1)				
				call send_new_computation(conf, conf%index_chunk, slave, packagesize)
				conf%index_chunk = conf%index_chunk + 1				
			enddo

		endif

		kill_flag = 0

! Work with the remaining chunks
		do while(kill_flag == 0)

!-------------------
! MASTER
!-------------------
			if (myrank == 0) then

! Receive spectrum
				call receive_spectrum(slave)

! If more chunks are available, send a new chunk to the inactive slave
				if (conf%index_chunk <= conf%nchunks) then					
					call send_new_computation(conf, conf%index_chunk, slave, packagesize)
					conf%index_chunk = conf%index_chunk + 1
				else

! If not, kill the slave
					write(*,FMT='(A,I3)') 'Master kills slave ', slave
					call kill_slave(slave)
					conf%slave_killed(slave) = 1			

! If no slave is active, active the kill_flag to exit the main loop
					if (sum(conf%slave_killed) == conf%nprocs-1) then
						kill_flag = 1
					endif
					
				endif
				
			endif

!-------------------
! SLAVE
!-------------------
			if (myrank /= 0) then

! Receive new chunk
				call receive_new_computation(spectrum%ini_lambda, spectrum%end_lambda, kill_flag, packagesize)

				if (kill_flag == 0) then

! Do the actual computation
					call setup_spectrum(conf, atm, spectrum)
					
 					call read_lines(atm, spectrum)

 					spectrum%flux = 0.d0

 					do i = 1, conf%nmus
						call add_background_opacity(atm, spectrum)
 						call add_line_opacity(atm, spectrum, conf%mus(i))
						call solve_rt(atm, spectrum, conf%mus(i), conf%weights(i), i, 1)

						call add_background_opacity(atm, spectrum)
						call solve_rt(atm, spectrum, conf%mus(i), conf%weights(i), i, 2)
					enddo

! Send back the spectrum
					call send_spectrum(myrank)

 					if (associated(spectrum%boundary)) deallocate(spectrum%boundary)
 					if (associated(spectrum%freq)) deallocate(spectrum%freq)
 					if (associated(spectrum%source)) deallocate(spectrum%source)
 					if (associated(spectrum%chi)) deallocate(spectrum%chi)
 					
 					if (associated(spectrum%flux)) deallocate(spectrum%flux)

 					if (index(atm%magnetic, 'NONMAGNETIC') == 0) then
						do i = 1, spectrum%nlines
							if (associated(spectrum%line(i)%splitting)) deallocate(spectrum%line(i)%splitting)
							if (associated(spectrum%line(i)%strength)) deallocate(spectrum%line(i)%strength)
						enddo
					endif
					if (associated(spectrum%line)) deallocate(spectrum%line)
				endif
				
			endif
			
		enddo

	else

! Serial mode
		do j = 1, conf%nchunks
		
			spectrum%ini_lambda = conf%chunks(1,j)
			spectrum%end_lambda = conf%chunks(2,j)

			call date_and_time(date, time, zone, values)
			write(*,FMT='(A,A,A,F8.2,A,F8.2,A)') '(t=',time,') * chunk [', conf%chunks(1,j), ',', conf%chunks(2,j), ']'
			
			call setup_spectrum(conf, atm, spectrum)

			call read_lines(atm, spectrum)

			spectrum%flux = 0.d0

			do i = 1, conf%nmus
				call add_background_opacity(atm, spectrum)
				call add_line_opacity(atm, spectrum, conf%mus(i))
				call solve_rt(atm, spectrum, conf%mus(i), conf%weights(i), i, 1)

				call add_background_opacity(atm, spectrum)
				call solve_rt(atm, spectrum, conf%mus(i), conf%weights(i), i, 2)
			enddo

			if (index(atm%magnetic, 'NONMAGNETIC') /= 0) then
				write(24) spectrum%nlambda, conf%nmus_intensity
				write(24) spectrum%freq, spectrum%flux(1,:,:)
			else
				write(24) spectrum%nlambda, conf%nmus_intensity
				write(24) spectrum%freq, spectrum%flux(:,:,:)
			endif
		enddo

	endif

	if (myrank == 0) then
		close(24)
	endif

! Finalize MPI
	call MPI_FINALIZE(mpi_status)
	
end program lte