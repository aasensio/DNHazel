! Select lines from the Kurucz linelist which are from the fundamental or first excited 
! ionization levels
program select
implicit none
	real(kind=8) :: wlength, lgf, elem, ioniz, isot_ab, k
	real(kind=8) :: j_low, e_low, e_up, j_up, gamma_rad, gamm_stark, vdw_damp, hyperf_str
	character(len=10) :: label_low, label_up
	character(len=4) :: reference
	character(len=1) :: hpflow_c, hpfup_c
	character(len=3) :: autoion
	integer :: nonlte_up, nonlte_low, isot, isot2, hpf_low, hpf_up, hpff_low, hpff_up, line_str
	integer :: i, j, lande_up, lande_low, isot_shift, nargs, nlines
	character(len=100) :: infile

		nargs = iargc()
		if (nargs == 1) then
	      call getarg(1,infile)
      endif

		open(unit=10,file=infile,status='old',action='read')

		k = 0
		do while (.not. eof(10))
			read(10,*)
			k = k + 1			
		enddo

		nlines = k

		rewind(10)

		open(unit=11,file=trim(adjustl(infile))//'.index',status='replace',action='write')
		k = 0
		do i = 1, nlines
			read(10,FMT='(F11.4,F7.3,F6.2,F12.3,F6.1,1X,A10,F12.3,F6.1,1X,A10,3F6.2,A4,2I2,I3,F6.3,I3,&
				&F6.3,2I5,1X,A1,A1,1X,A1,A1,i1,A3,2I5,I6)') wlength, lgf, elem, e_low, j_low, label_low,&
				e_up, j_up, label_up, gamma_rad, gamm_stark, vdw_damp, reference, nonlte_up,&
				nonlte_low, isot, hyperf_str, isot2, isot_ab, hpf_low, hpf_up, hpff_low,&
				hpflow_c, hpff_up, hpfup_c, line_str, autoion, lande_up, lande_low, isot_shift
			wlength = wlength * 10.d0
			
			if (wlength > k) then				
				write(11,FMT='(F10.2,3X,I6)') k, i
				k = k + 100.d0
			endif
			
		enddo
		print *, 'Total number of lines = ', nlines
		close(10)
		close(11)

end program select
