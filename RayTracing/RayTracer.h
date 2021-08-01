#pragma once

#include "cuda_runtime.h"

#include <GL/glew.h>
#include <GL/GL.h>

#include "glm/vec3.hpp"
#include "glm/vec4.hpp"
#include "glm/mat4x4.hpp"
#include "glm/ext/matrix_transform.hpp"
#include "glm/ext.hpp"					

#include "Scene.h"

using namespace glm;

#define TEX_WIDTH 1024
#define TEX_HEIGHT 512

#define MAX_BOUNCE 3

class RayTracer
{
public:
	Sphere m_GlasSphere;
	vec4 m_GlassSphereVelocity;
	vec3 m_Camera;
	GLfloat *m_Texels;
	float *d_Texels;

	RayTracer();
	~RayTracer();
	void init();
	void render(GLuint textureID, bool bCudaOn, bool bShadowOn, bool gAntiAliasingOn);
};

class Ray
{
public:
	vec3 origin;
	vec3 dir;

	__device__ __host__ Ray(vec3 orig, vec3 direction) : origin(orig), dir(direction) {};
	__device__ __host__ vec3 getIntersectionPoint(float T) { return (origin + T * dir); };
};

enum ObjectType
{
	NONE = 0,
	CUBE,
	SPHERE,
	QUAD,
};

// Holds the information of a ray's intersection with an object
struct HitInfo
{
	bool hit;
	float reflectance;
	float paramT;
	ObjectType objType;
	int index;
};
