#pragma once

#include "glm/vec3.hpp"
#include "glm/vec4.hpp"

using namespace glm;

#ifndef RT_SCENE_HEADER
#define RT_SCENE_HEADER

#define NUM_CUBES 2
#define NUM_SPHERES 3
#define NUM_QUADS 1

struct Cube
{
	vec4 min;
	vec4 max;
	vec4 normal;
};

struct Sphere
{
	vec4 pos;
};

struct Triangle
{
	vec4 a, b, c;
	vec4 normal;
};

struct Quad
{
	vec4 a, b, c, d;
	vec4 normal;
};

struct TexQuad
{
	Quad q;
	int texIndex;
};

struct Material
{
	vec4 ambient, diffuse, specular;
	float shininess;
	float reflectance;
};

struct Light
{
	vec4 position;
	vec4 color;
	float brightness;
};

extern Cube cubes[];
extern Material cube_materials[];

extern Sphere spheres[];
extern Material sphere_materials[];

extern Quad quads[];
extern Material quad_materials[];

extern TexQuad texQuads[];
extern int num_texQuads;

extern Light Light1;

extern Sphere GlassSphere;
extern Material GlassSphereMaterial;
extern vec3 GlassSphereVelocity;

#endif