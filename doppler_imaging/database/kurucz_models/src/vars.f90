module vars_mod
use types_mod
implicit none

	real(kind=8), parameter :: PK = 1.3806503d-16, UMA = 1.66053873d-24, PC = 2.99792458d10
	real(kind=8), parameter :: PH = 6.62606876d-27, PHK = PH / PK, PHC = 2.d0 * PH / PC**2
	real(kind=8), parameter :: PME = 9.10938188d-28, PE = 4.8032d-10, PI = 3.14159265359d0
	real(kind=8), parameter :: PHC2 = 2.d0 * PH * PC**2
	real(kind=8), parameter :: SQRTPI = 1.77245385091d0, EARTH_SUN_DISTANCE = 1.495979d13
	real(kind=8), parameter :: LARMOR = PE/(4.d0*PI*PME*PC), PMP = 1.67262158d-24
	real(kind=8), parameter :: NUCLEAR_MAGNETON = PE*PH/(2.d0*PI*PMP)
	real(kind=8), parameter :: BOHR_MAGNETON = PE*PH/(2.d0*PI*PME)
	real(kind=8), parameter :: PH2C3 = 2.d0 * PH**2 * PC**3, PERTURBATION = 0.005d0
	real(kind=8), parameter :: OPA = PI * PE**2 / (PME * PC), EV_ERG = 1.60217646d-12
	
	type(atmosphere_type) :: atm

	type(config_type) :: conf

	type(spectrum_type) :: spectrum

	character(len=100) :: config_file

	integer :: myrank

	character(len=3) :: ionization_state(3) = (/'I  ','II ','III'/)

	real(kind=8) :: identity(4,4)

	real(kind=8) :: fact(0:301)
	
	integer, parameter :: nMolecules = 4
	integer, parameter :: mol_code(nMolecules) = (/15, 31, 13, 107/)     ! Indices of the molecules in the chemical equilibrium
	
end module vars_mod