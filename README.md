# Building #
Build on the GHC machines containing NVIDIA GeForce RTX 2080 B GPUs. These have host names ```ghcX.ghc.andrew.cmu.edu```, for ```X``` between 47 and 86. To compile, run the following commands in the same directory as ```CMakeLists.txt```:
```
mkdir build
cd build
cmake .. "-DCMAKE_PREFIX_PATH=lib;/usr/local/cuda-11.7/lib64"
make
```

# Credits #
TXT by HTML5 UP
html5up.net | @ajlkn
Free for personal and commercial use under the CCA 3.0 license (html5up.net/license)

Demo Images:
- Unsplash (unsplash.com)

Icons:
- Font Awesome (fontawesome.io)

Other:
- jQuery (jquery.com)
- Responsive Tools (github.com/ajlkn/responsive-tools)
