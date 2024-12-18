#######################################
# racha-ThinkPad-A475.mk
# Default options for racha-ThinkPad-A475 computer
#######################################
CC = gcc
LIBSLOCAL = -L/usr/lib -llapack -lblas -lm
INCLUDEBLASLOCAL = -I/usr/include
OPTCLOCAL = -fPIC -march=native
