### A simple Ray Tracer created using CUDA and OpenGL.

- This program creates a texture where each pixel of the texture corresponds to one ray. Tracing the rays is parallelized using CUDA kernels. Each thread traces one ray. The threads are arranged in blocks of 16x16 and the number of blocks is determined by texture width and height.
- A sequential (CPU) implementation of the ray tracer is also included. One can switch between CPU and GPU implementation on the fly to compare the results.



### Screenshots


![](/Screenshot_1.png)

![](/Screenshot_2.png)