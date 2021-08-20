# Copyright (c) 1993-2015, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

reduce: mainTemplate.cu *.h
	hipcc -O3 -std=c++17 -DDATATYPE=int -DVECTYPE=int2 mainTemplate.cu -o reduceInt.x -I/opt/rocm-4.3.0/hip/include/hip/amd_detail -I/nobackup/HOME/dkokron/code-samples/posts/parallel_reduction_with_shfl/hipCUB/install/hipcub/include -I/nobackup/HOME/dkokron/code-samples/posts/parallel_reduction_with_shfl/hipCUB/build/rocprim-src/rocprim/include
	hipcc -O3 -std=c++17 -DDATATYPE=float -DVECTYPE=float2 mainTemplate.cu -o reduceFloat.x -I/opt/rocm-4.3.0/hip/include/hip/amd_detail -I/nobackup/HOME/dkokron/code-samples/posts/parallel_reduction_with_shfl/hipCUB/install/hipcub/include -I/nobackup/HOME/dkokron/code-samples/posts/parallel_reduction_with_shfl/hipCUB/build/rocprim-src/rocprim/include
	hipcc -O3 -std=c++17 -DDATATYPE=double -DVECTYPE=double2 mainTemplate.cu -o reduceDouble.x -I/opt/rocm-4.3.0/hip/include/hip/amd_detail -I/nobackup/HOME/dkokron/code-samples/posts/parallel_reduction_with_shfl/hipCUB/install/hipcub/include -I/nobackup/HOME/dkokron/code-samples/posts/parallel_reduction_with_shfl/hipCUB/build/rocprim-src/rocprim/include

clean:
	rm -f reduceInt.x reduceFloat.x reduceDouble.x

