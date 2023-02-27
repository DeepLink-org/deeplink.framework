#include <iostream>
#include <stdio.h>
#include <stdlib.h>


#include <string>
#include <algorithm>
#include <vector>
#include <functional>
#include <unordered_set>

#include <cmath>
#include <math.h>
#include <numeric>
#include <fcntl.h>
#include <errno.h>
#include <time.h>
#include <unistd.h>
#include <typeinfo>
#include <assert.h>


void init_cuda()
{
        int count;
        cudaGetDeviceCount(&count);
        int i;
        for (i = 0; i < count; i++) {
                cudaDeviceProp prop;
                if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
                        if (prop.major >= 1) {
                                printf("gpu rate %d \n", prop.clockRate * 1000);
                                break;
                        }
                }
        }
        cudaSetDevice(i);
}

