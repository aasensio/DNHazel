pro generateModel
	Teff = 5000.
	gravity = 4.0
	feh = 0.0
	out = 'output'
	kmod, Teff, gravity, feh, out  ;<- extracts a model from the Kurucz database

	B = replicate(10.d0, 72)
	thB = replicate(0.d0, 72)
	phiB = replicate(0.d0, 72)
	BField = transpose([[B],[thB],[phiB]])
	genmod, out, 'T5000_logg4.0_feh0.0_10G.model', BField=BField  ; <- generate the final model including field (if BField does not appear, it generates a non-magnetic model)
end