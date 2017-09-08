

Para cargar cada perfil:

data = np.load('pix__XX.npy',encoding = 'latin1').item()
modelo = data['modelos']
perfiles = data['perfiles'] 
nite = modelo.shape[0]

perfilObservacional = perfiles[iteracion,0,lambda,stokes]
perfilesSynteticos  = perfiles[iteraciones,1,lambda,stokes]
modelos = modelo[iteracion,magnitud,tau]