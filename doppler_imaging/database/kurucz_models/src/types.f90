module types_mod
implicit none

	type atmosphere_type
		character(len=15) :: hscale, magnetic
		integer :: n_depths, direction
		real(kind=8) :: avgWeight
		real(kind=8), dimension(:), pointer :: height, cmass, T, microturb, vmacro, ne, mu_angles, Pe
		real(kind=8), dimension(:), pointer :: B, thetaB, chiB
		real(kind=8), dimension(:), pointer :: cos_gamma, sin_gamma, cos_2chi, sin_2chi
		real(kind=8), dimension(:), pointer :: weights
		real(kind=8), dimension(:), pointer :: abundances, mass
		real(kind=8), dimension(:,:), pointer :: doppler_width, abund2, partition, ionization_pot, mol_density, partition_functions_molecular
		real(kind=8), dimension(:), pointer :: PH, PHminus, PHplus, PH2, PH2plus, P_total, nHtot
		real(kind=8), dimension(:,:,:), pointer :: partition_functions
		character(len=2), dimension(:), pointer :: element_symbol
	end type atmosphere_type

	type atomic_transition_type
		real(kind=8) :: wlength, gf, expot, freq, lande_up, lande_low, J_up, J_low
		integer :: element, ionization, mass, molecule
! Mlow=Mup-1 q=1
! Mlow=Mup   q=2
! Mlow=Mup+1 q=3
		real(kind=8), dimension(:,:), pointer :: strength(:,:), splitting(:,:)
		integer :: n_components(3)
		logical :: el_dipole, ABO_parameters_present
		real(kind=8) :: gamma_vdw, gamma_stark, gamma_rad, enhacement_vdw, sigma0_ABO, alpha_ABO
		integer :: lambda_from, lambda_to, atomic_molecular
	end type atomic_transition_type

	type spectrum_type
		integer :: nlambda, nlines
		real(kind=8) :: ini_lambda, end_lambda, step_lambda
		real(kind=8), pointer :: freq(:), flux(:,:,:)
		real(kind=8), pointer :: chi(:,:,:), source(:,:), boundary(:,:)
		type(atomic_transition_type), pointer :: line(:)
	end type spectrum_type

	type config_type
		character(len=10) :: mode
		integer :: nmus, nprocs, nchunks, index_chunk, nmus_intensity
		real(kind=8), pointer :: mus(:), chunks(:,:), weights(:)
		character(len=100) :: atm_file, out_file, linelist_file
		real(kind=8) :: ini_lambda, end_lambda, step_lambda, chunk_size
		integer, pointer :: slave_killed(:)
	end type config_type

end module types_mod