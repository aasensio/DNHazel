; Transform all models from tau axis to z axis
pro transform
	f = file_search('*.mod')
	for i = 0, n_elements(f)-1 do begin
		a = ddread(f[i],offset=1,/count)
		n = n_elements(a[0,*])
		b = fltarr(8,n)
		
; Transform tau-axis to z-axis
		tau = 10.d0^a[0,*]
		T = a[1,*]
		Pe = a[2,*]
		vmic = a[3,*]
		B = a[4,*]		
		vlos = a[5,*]
		inclination = a[6,*]
		azimuth = a[7,*]
		n_e = Pe / (1.381d-16*T)
		z = tau
		z[0] = 0.d0
		opa500 = fltarr(n)
		for j = 0, n-1 do begin
			res = kappa_c(Pe[j],T[j],5000.0,htoverv)
			opa500[j] = res * htoverv
		endfor
		
		for j = 1, n-1 do begin
			z[j] = z[j-1] - 2.0*(tau[j]-tau[j-1]) / (opa500[j] + opa500[j-1])
		endfor
		
		hZero = interpol(z, tau, 1.0)
		
		z = z - hZero
		
		B = replicate(2000.d0,n)
		inclination = replicate(20.d0,n)
		azimuth = fltarr(n)
		print, 'Writing file : ', f[i]+'.atmos'
		openw,2,f[i]+'.atmos',width=132
		printf, 2, '# Model  x [km]   T[K]     ne [cm^-3]    vturb [cm/s]       B[G]       thetaB[deg]       chiB[deg]'
		printf,2,"'Z'"
		printf,2,"'MAGNETIC'"
		printf,2,"NORMAL_ABUNDANCES"
		printf,2,n

		for j = 0, n-1 do begin
			printf,2,z[j] / 1d5,T[j],n_e[j],vmic[j]*1d-5,vlos[j]*1d-5,B[j],inclination[j],azimuth[j]
		endfor
		close,2
				
	endfor
	
end
