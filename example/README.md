# Running the example

* Install KaHIP following the instructions in https://github.com/schulzchristian/KaHIP
* Export the kahip root folder that contains deploy (eg. `$export KAHIP_ROOT= /usr/local/share/KaHIP`)
* Then run:
```
mkdir build
cd build
cmake ..
make
mpiexec -n 4 ./ex
```
