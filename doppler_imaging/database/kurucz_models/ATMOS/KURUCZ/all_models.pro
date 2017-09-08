pro all_models
	T = 3500 + 250 * findgen(27)
	for i = 0, 26 do begin
		Teff = T[i]
        	gravity = 4.0
        	feh = 0.0
        	out = 'output'
        	kmod, Teff, gravity, feh, out  ;<- extracts a model from the Kurucz database

        	B = replicate(0.d0, 72)
        	thB = replicate(0.d0, 72)
        	phiB = replicate(0.d0, 72)
        	BField = transpose([[B],[thB],[phiB]])
        	genmod, out, 'T_'+strtrim(string(fix(Teff)),2)+'_logg4.0_feh0.0.model'  ; <- generate the final model including field (if BField does not appear, it generates a non-magnetic model)
	endfor
end
