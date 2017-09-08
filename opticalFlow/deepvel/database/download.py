
import shutil
import urllib.request

# Download the file from `url` and save it locally under `file_name`:

for i in range(250):

    url = "http://www.lmsal.com/~cheung/muram/ar21200_lambda0.5/I_out.{0:06d}".format(i*1000)
    file_name = "/scratch1/3dcubes/cheung/I_out.{0:06d}".format(i*1000)
    print("Downloading {0}".format(file_name))
    with urllib.request.urlopen(url) as response, open(file_name, 'wb') as out_file:
        shutil.copyfileobj(response, out_file)

for i in range(250):
    url = "http://www.lmsal.com/~cheung/muram/ar21200_lambda0.5/tau_slice_0.100.{0:06d}".format(i*1000)
    file_name = "/scratch1/3dcubes/cheung/tau_slice_0.100.{0:06d}".format(i*1000)
    print("Downloading {0}".format(file_name))
    with urllib.request.urlopen(url) as response, open(file_name, 'wb') as out_file:
        shutil.copyfileobj(response, out_file)
