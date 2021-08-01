#include <stdio.h>
#include <Windows.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "RayTracer.h"

using namespace glm;

void HandleCudaError(cudaError_t error, const char* file, int line)
{
	if (error != cudaSuccess)
	{
		const char* errorString = cudaGetErrorString(error);
		char errorMsg[256];
		sprintf(errorMsg, "Error \"%s\" at file %s line %d", errorString, file, line);
		MessageBox(NULL, errorMsg, TEXT("Error"), MB_OK);
		exit(EXIT_FAILURE);
	}
}

#define HANDLE_ERROR(err) (HandleCudaError(err, __FILE__, __LINE__))

RayTracer RT;

__constant__ Cube d_Cubes[NUM_CUBES];
__constant__ Material d_CubeMaterials[NUM_CUBES];

__constant__ Sphere d_Spheres[NUM_SPHERES];
__constant__ Material d_SphereMaterials[NUM_SPHERES];

__constant__ Quad d_Quads[NUM_QUADS];
__constant__ Material d_QuadMaterials[NUM_QUADS];

__constant__ Light d_Light1;
__constant__ vec3 d_Camera;
__constant__ bool d_bShadowOn;
__constant__ bool d_bAntiAliasingOn;

RayTracer::RayTracer()
{
}

RayTracer::~RayTracer()
{
	cudaFreeHost(m_Texels);
}

void RayTracer::init()
{
	m_Camera = vec3(0.0f, 0.0f, -3.0f);
	
	// Allocate pinned memory on which the kernel will write pixel data
	HANDLE_ERROR(cudaHostAlloc((void**)&m_Texels, TEX_WIDTH * TEX_HEIGHT * 4 * sizeof(GLfloat), cudaHostAllocMapped));
	HANDLE_ERROR(cudaHostGetDevicePointer(&d_Texels, m_Texels, 0));

	// Copy object and material data to device memory
	// Cube
	HANDLE_ERROR(cudaMemcpyToSymbol(d_Cubes, cubes, NUM_CUBES * sizeof(Cube), 0, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpyToSymbol(d_CubeMaterials, cube_materials, NUM_CUBES * sizeof(Material), 0, cudaMemcpyHostToDevice));

	// Sphere
	HANDLE_ERROR(cudaMemcpyToSymbol(d_SphereMaterials, sphere_materials, NUM_SPHERES * sizeof(Material), 0, cudaMemcpyHostToDevice));

	// Quad
	HANDLE_ERROR(cudaMemcpyToSymbol(d_Quads, quads, NUM_QUADS * sizeof(Quad), 0, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpyToSymbol(d_QuadMaterials, quad_materials, NUM_QUADS * sizeof(Material), 0, cudaMemcpyHostToDevice));

	// Camera
	HANDLE_ERROR(cudaMemcpyToSymbol(d_Camera, &m_Camera, sizeof(m_Camera), 0, cudaMemcpyHostToDevice));

	// Light
	HANDLE_ERROR(cudaMemcpyToSymbol(d_Light1, &Light1, sizeof(Light), 0, cudaMemcpyHostToDevice));
}

// Initialize a ray starting from camera to pixel px
__device__ Ray cudaInitRay(vec2 px)
{
	float NDCx = (px.x + 0.5f) / float(TEX_WIDTH);
	float NDCy = (px.y + 0.5f) / float(TEX_HEIGHT);

	float x = 2.0f * NDCx - 1.0f;
	float y = 2.0f * NDCy - 1.0f;

	y = y * float(TEX_HEIGHT) / float(TEX_WIDTH); // Multiply y by screen's aspect ratio

	vec4 pixel = vec4(d_Camera, 0.0f) + vec4(x, y, 1.5f, 0.0f);
	vec3 direction = normalize(vec3(pixel.x - d_Camera.x, pixel.y - d_Camera.y, pixel.z - d_Camera.z));

	return Ray(d_Camera, direction);
}

__device__ HitInfo cudaIntersectTriangle(Ray r, Triangle T)
{
	HitInfo i;
	i.hit = false;

	vec3 AB = T.b - T.a;
	vec3 AC = T.c - T.a;
	mat3 mat = mat3(AB, AC, -1.0f * r.dir);

	float d = determinant(mat);
	if (d != 0.0f)
	{
		vec3 OA = r.origin - vec3(T.a);
		mat3 inv = inverse(mat);
		vec3 sol = inv * OA;

		if (sol.x >= -0.0001f && sol.x <= 1.0001f)
		{
			if (sol.y >= -0.0001f && sol.y <= 1.0001f)
			{
				if ((sol.x + sol.y <= 1.0001f) && sol.z >= -0.0001f)
				{
					i.hit = true;
					i.paramT = sol.z;
				}
			}
		}
	}
	return i;
}

__device__ HitInfo cudaIntersectQuad(Ray r, Quad Q)
{
	Triangle T1 = { Q.a, Q.b, Q.c, Q.normal };
	HitInfo i = cudaIntersectTriangle(r, T1);
	if (i.hit == false)
	{
		Triangle T2 = { Q.c, Q.d, Q.a, Q.normal };
		i = cudaIntersectTriangle(r, T2);
	}
	return i;
}

// Check if the ray intersects with any of Quads
__device__ HitInfo cudaCheckQuads(Ray r, int skipIndex)
{
	HitInfo hitInfo, tempInfo;
	hitInfo.hit = false;
	float T = 100.0f, temp_param;

	for (int i = 0; i < NUM_QUADS; i++)
	{
		if (i != skipIndex)
		{
			tempInfo = cudaIntersectQuad(r, d_Quads[i]);
			if (tempInfo.hit == true)
			{
				temp_param = tempInfo.paramT;
				if (temp_param < T)
				{
					T = temp_param;
					tempInfo.objType = QUAD;
					tempInfo.index = i;
					hitInfo = tempInfo;
					hitInfo.reflectance = d_QuadMaterials[i].reflectance;
				}
			}
		}
	}
	return hitInfo;
}

__device__ HitInfo cudaIntersectSphere(Ray r, Sphere s)
{
	HitInfo hitInfo;
	hitInfo.hit = false;

	vec3 oc = r.origin - vec3(s.pos);
	float a = dot(r.dir, r.dir);
	float b = 2 * dot(oc, r.dir);
	float c = dot(oc, oc) - s.pos.w * s.pos.w;

	float d = b * b - 4 * a * c;

	if (d >= 0.0f)
	{
		float t1, t2;

		t1 = (-b - sqrt(d)) / (2.0f * a);
		t2 = (-b + sqrt(d)) / (2.0f * a);

		if (t1 > 0.0f && t2 > 0.0f && t1 < t2)
		{
			hitInfo.hit = true;
			hitInfo.paramT = t1;
		}
	}
	return hitInfo;
}

// Check if the ray intersects with any of Spheres
__device__ HitInfo cudaCheckSpheres(Ray r, int skipIndex)
{
	HitInfo hitInfo, tempInfo;
	hitInfo.hit = false;
	float T = 100.0f, temp_param;

	for (int i = 0; i < NUM_SPHERES; i++)
	{
		if (i != skipIndex)
		{
			tempInfo = cudaIntersectSphere(r, d_Spheres[i]);
			if (tempInfo.hit == true)
			{
				temp_param = tempInfo.paramT;
				if (temp_param < T)
				{
					T = temp_param;
					tempInfo.objType = SPHERE;
					tempInfo.index = i;
					hitInfo = tempInfo;
					hitInfo.reflectance = d_SphereMaterials[i].reflectance;
				}
			}
		}
	}
	return hitInfo;
}

__device__ HitInfo cudaIntersectCube(Ray ray, Cube c)
{
	HitInfo i;
	i.hit = false;

	vec3 tMin = (vec3(c.min) - ray.origin) / ray.dir;
	vec3 tMax = (vec3(c.max) - ray.origin) / ray.dir;
	vec3 t1 = min(tMin, tMax);
	vec3 t2 = max(tMin, tMax);

	float tNear = max(max(t1.x, t1.y), t1.z);
	float tFar = min(min(t2.x, t2.y), t2.z);

	if (tNear > 0.0f && tNear < tFar)
	{
		i.hit = true;
		i.paramT = tNear;
	}
	return i;
}

// Check if the ray intersects with any of Cubes
__device__ HitInfo cudaCheckCubes(Ray r, int skipIndex)
{
	HitInfo hitInfo, tempInfo;
	hitInfo.hit = false;
	float T = 100.0f, temp_param;

	for (int i = 0; i < NUM_CUBES; i++)
	{
		if (i != skipIndex)
		{
			tempInfo = cudaIntersectCube(r, d_Cubes[i]);
			if (tempInfo.hit == true)
			{
				temp_param = tempInfo.paramT;
				if (temp_param < T)
				{
					T = temp_param;
					tempInfo.objType = CUBE;
					tempInfo.index = i;
					hitInfo = tempInfo;
					hitInfo.reflectance = d_CubeMaterials[i].reflectance;
				}
			}
		}
	}
	return hitInfo;
};

// Get the first intersection of a ray among all objects
__device__ HitInfo cudaGetFirstIntersection(Ray r, int skipType, int skipIndex)
{
	HitInfo hiCube, hiSphere, hiQuad, hiFinal;
	hiFinal.hit = false;

	int skip[5] = { -1, -1, -1, -1 };
	skip[skipType] = skipIndex;

	float t1 = 100.0f, t2 = 100.0f;

	hiCube = cudaCheckCubes(r, skip[1]);
	if (hiCube.hit == true)
	{
		t2 = hiCube.paramT;
		if (t2 < t1)
		{
			t1 = t2;
			hiFinal = hiCube;
		}
	}

	hiSphere = cudaCheckSpheres(r, skip[2]);
	if (hiSphere.hit == true)
	{
		t2 = hiSphere.paramT;
		if (t2 < t1)
		{
			t1 = t2;
			hiFinal = hiSphere;
		}
	}

	hiQuad = cudaCheckQuads(r, skip[3]);
	if (hiQuad.hit == true)
	{
		t2 = hiQuad.paramT;
		if (t2 < t1)
		{
			t1 = t2;
			hiFinal = hiQuad;
		}
	}

	return hiFinal;
}

/****************** Color Calculations ***************/

__device__ vec4 cudaPhong(Ray r, HitInfo hi, vec3 normal, Material mtl)
{
	vec3 ip = r.getIntersectionPoint(hi.paramT);
	vec3 normalized_light_direction = normalize(vec3(d_Light1.position) - ip);

	float dist = length(ip - vec3(d_Light1.position));
	float attn = (1 / (1 + 0.0001f * dist + 0.000002f * dist * dist));

	vec4 ambient = mtl.ambient * 0.2f;

	float tn_dot_ld = max(dot(normal, normalized_light_direction), 0.0f);
	vec4 diffuse = d_Light1.color * mtl.diffuse * tn_dot_ld * attn;

	vec3 reflection_vector = reflect(normalized_light_direction, normal);
	vec4 specular = mtl.specular * pow(max(dot(reflection_vector, r.dir), 0.0f), mtl.shininess);

	vec4 color = (ambient + diffuse + specular);
	return  (1 - mtl.reflectance) * color;
}

__device__ vec4 cudaTraceShadowRay(Ray r, HitInfo hi, vec3 normal, Material mtl)
{
	vec3 ip = r.getIntersectionPoint(hi.paramT);
	vec3 direction = normalize(vec3(d_Light1.position) - ip);
	Ray shadowRay = Ray(ip, direction);

	if (d_bShadowOn)
	{
		HitInfo shadow_info = cudaGetFirstIntersection(shadowRay, hi.objType, hi.index);

		if (shadow_info.hit == false)
		{
			return cudaPhong(r, hi, normal, mtl);
		}
		else
		{
			return (1 - mtl.reflectance) * (mtl.ambient * 0.5f);
		}
	}
	else
		return cudaPhong(r, hi, normal, mtl);
}

__device__ vec4 cudaCalculateColor(Ray r, HitInfo hi)
{
	vec4 color = vec4(0.0f);
	Material mtl;
	vec3 normal;
	switch (hi.objType)
	{
	case CUBE:
		normal = vec3(d_Cubes[hi.index].normal);
		mtl = d_CubeMaterials[hi.index];
		break;
	case SPHERE:
		vec3 ip = r.getIntersectionPoint(hi.paramT);
		normal = normalize(ip - vec3(d_Spheres[hi.index].pos));
		mtl = d_SphereMaterials[hi.index];
		break;
	case QUAD:
		normal = vec3(d_Quads[hi.index].normal);
		mtl = d_QuadMaterials[hi.index];
		break;
	}

	return cudaTraceShadowRay(r, hi, normal, mtl);
}

__device__ Ray cudaGetReflectedRay(Ray r, HitInfo hi)
{
	vec3 intersection_point = r.getIntersectionPoint(hi.paramT);
	vec3 normal;

	switch (hi.objType)
	{
	case SPHERE:
		normal = normalize(intersection_point - vec3(d_Spheres[hi.index].pos));
		break;
	case CUBE:
		normal = vec3(d_Cubes[hi.index].normal);
		break;
	case QUAD:
		normal = vec3(d_Quads[hi.index].normal);
		break;
	}
	vec3 reflect_dir = reflect(r.dir, normal);

	return Ray(intersection_point, reflect_dir);
};

__device__ vec4 cudaTrace(Ray r)
{
	HitInfo info;
	vec4 color = vec4(0.0f), temp_color;
	float last_reflectance = 1.0f;
	int skipType = NONE, skipIndex = -1;

	for (int i = 0; i <= MAX_BOUNCE; i++)
	{
		info = cudaGetFirstIntersection(r, skipType, skipIndex);
		if (info.hit == true)
		{
			temp_color = cudaCalculateColor(r, info);
			color = color + last_reflectance * temp_color;

			if (info.reflectance > 0.0f)
			{
				r = cudaGetReflectedRay(r, info);
				last_reflectance = info.reflectance;
			}
			else
				break;
		}
		else
		{
			color += last_reflectance * vec4(0.2f, 0.2f, 0.5f, 0.0f);
			break;
		}

		skipType = info.objType;
		skipIndex = info.index;
	}
	return color;
}

// Tracer for anti aliasing
__global__ void cudaTraceAA(vec2* childCoord, vec4* colors)
{
	int tid = threadIdx.x;
	vec2 pixel;

	pixel.x = childCoord[tid].x;
	pixel.y = childCoord[tid].y;

	Ray ray = cudaInitRay(pixel);
	colors[tid] = cudaTrace(ray);
}

__global__ void cudaRayTracerStart(float* Texels)
{
	int tid_x = blockDim.x * blockIdx.x + threadIdx.x;
	int tid_y = blockDim.y * blockIdx.y + threadIdx.y;

	if (tid_x < TEX_WIDTH && tid_y < TEX_HEIGHT)
	{
		vec2 pixel = vec2(tid_x, tid_y);
		vec4 color;

		if (d_bAntiAliasingOn == true)
		{
			// Create four subrays in four corners of the pixel
			// Trace all of them and take average of their color values and assign it the pixel

			vec2* d_ChildCoord;
			vec4* childColors;

			vec2 childCoord[4];
			childCoord[0].x = tid_x - 0.25f;
			childCoord[0].y = tid_y + 0.25f;

			childCoord[1].x = tid_x + 0.25f;
			childCoord[1].y = tid_y + 0.25f;

			childCoord[2].x = tid_x + 0.25f;
			childCoord[2].y = tid_y - 0.25f;

			childCoord[3].x = tid_x - 0.25f;
			childCoord[3].y = tid_y - 0.25f;

			cudaMalloc((void**)&d_ChildCoord, sizeof(childCoord));
			memcpy(d_ChildCoord, &childCoord, sizeof(childCoord));
			cudaMalloc((void**)&childColors, sizeof(vec4) * 4);

			// Recursively launch the kernel for child rays
			cudaTraceAA << <1, 4 >> > (d_ChildCoord, childColors);
			cudaDeviceSynchronize();

			color = (childColors[0] + childColors[1] + childColors[2] + childColors[3]) / 4.0f;

			cudaFree(childColors);
			cudaFree(d_ChildCoord);

			/*Ray r1 = cudaInitRay(vec2(pixel.x + 0.25f, pixel.y + 0.25f));
			Ray r2 = cudaInitRay(vec2(pixel.x - 0.25f, pixel.y + 0.25f));
			Ray r3 = cudaInitRay(vec2(pixel.x + 0.25f, pixel.y - 0.25f));
			Ray r4 = cudaInitRay(vec2(pixel.x - 0.25f, pixel.y - 0.25f));

			vec4 c1 = cudaTrace(r1);
			vec4 c2 = cudaTrace(r2);
			vec4 c3 = cudaTrace(r3);
			vec4 c4 = cudaTrace(r4);

			color = (c1 + c2 + c3 + c4) / 4.0f;*/
		}
		else
		{
			Ray ray = cudaInitRay(pixel);
			color = cudaTrace(ray);
		}

		int index = (tid_x * TEX_HEIGHT + tid_y) * 4;

		Texels[index] = color.r;
		Texels[index + 1] = color.g;
		Texels[index + 2] = color.b;
		Texels[index + 3] = 1.0f;
	}
}

float Sphere0Velocity = -0.02f;
float Sphere1Velocity = -0.02f;

void RayTracer::render(GLuint textureID, bool bCudaOn, bool bShadowOn, bool bAntiAliasingOn)
{
	vec4 RayTracerStart(int, int, vec3, bool);
	void RTCudaMain();

	// Update the sphere positions according to their velocity
	spheres[0].pos.y += Sphere0Velocity;
	spheres[1].pos.y += Sphere1Velocity;
	if (spheres[0].pos.y < -0.3f || spheres[0].pos.y > 1.0f)
	{
		Sphere0Velocity = -Sphere0Velocity;
	}
	if (spheres[1].pos.y < -0.3f || spheres[1].pos.y > 1.0f)
	{
		Sphere1Velocity = -Sphere1Velocity;
	}

	if (bCudaOn)
	{
		dim3 blocks(TEX_WIDTH / 16, TEX_HEIGHT / 16);
		dim3 threads(16, 16);

		// Following data can change between kernel calls. Hence copy them just before launching the kernel
		HANDLE_ERROR(cudaMemcpyToSymbol(d_bShadowOn, &bShadowOn, sizeof(bool), 0, cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpyToSymbol(d_bAntiAliasingOn, &bAntiAliasingOn, sizeof(bool), 0, cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpyToSymbol(d_Spheres, spheres, NUM_SPHERES * sizeof(Sphere), 0, cudaMemcpyHostToDevice));

		cudaRayTracerStart << <blocks, threads >> > (d_Texels);
		HANDLE_ERROR(cudaGetLastError());

	}
	else
	{
		for (int x = 0; x < TEX_WIDTH; x++)
		{
			for (int y = 0; y < TEX_HEIGHT; y++)
			{
				vec4 texelColor = RayTracerStart(x, y, m_Camera, bAntiAliasingOn);

				int index = (x * TEX_HEIGHT + y) * 4;
				m_Texels[index] = texelColor.r;
				m_Texels[index + 1] = texelColor.g;
				m_Texels[index + 2] = texelColor.b;
				m_Texels[index + 3] = 1.0f;
			}
		}
	}

	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

	glBindTexture(GL_TEXTURE_2D, textureID);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, TEX_WIDTH, TEX_HEIGHT, 0, GL_RGBA, GL_FLOAT, m_Texels);

	glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
}