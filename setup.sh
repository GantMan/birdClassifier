# make sure python 3.6.5 is installed, 
# tensorflow not supporting 3.7 as of this code
# brew unlink python # If you have installed (with brew) another version of python
# brew install https://raw.githubusercontent.com/Homebrew/homebrew-core/f2a764ef944b1080be64bd88dca9a1d80130c558/Formula/python.rb

# install keras/tensor stuffs
pip install keras 
pip install tensorflow

# install cuda
# https://developer.nvidia.com/cuda-downloads
# pip install tensorflow-gpu

# We need numpy (error on missing numpy is crazy)
brew install numpy

# basic python stuff
pip install pillow
pip install sklearn