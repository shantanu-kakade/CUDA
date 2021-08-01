#include "glm/mat4x4.hpp"
#include "glm/ext/matrix_transform.hpp"	
#include "glm/ext.hpp"					

#include "Scene.h"
#include "RayTracer.h"

/*	This files contains the sequential (CPU) implmentation of ray tracing
*   The code is same as the CUDA kernal
*/
Ray initRay(vec2 px, vec3 camera)
{
	float NDCx = (px.x + 0.5f) / float(TEX_WIDTH);
	float NDCy = (px.y + 0.5f) / float(TEX_HEIGHT);

	float x = 2.0f * NDCx - 1.0f;
	float y = 2.0f * NDCy - 1.0f;

	y = y * float(TEX_HEIGHT) / float(TEX_WIDTH); // Multiply y by screen's aspect ratio

	vec4 pixel = vec4(camera, 0.0f) + vec4(x, y, 1.5f, 0.0f);

	vec3 direction = normalize(vec3(pixel.x - camera.x, pixel.y - camera.y, pixel.z - camera.z));

	return Ray(camera, direction);
}

HitInfo intersectTriangle(Ray r, Triangle T)
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

HitInfo intersectQuad(Ray r, Quad Q)
{
	Triangle T1 = { Q.a, Q.b, Q.c, Q.normal };
	HitInfo i = intersectTriangle(r, T1);
	if (i.hit == false)
	{
		Triangle T2 = { Q.c, Q.d, Q.a, Q.normal };
		i = intersectTriangle(r, T2);
	}
	return i;
}

HitInfo CheckQuads(Ray r, int skipIndex)
{
	HitInfo hitInfo, tempInfo;
	hitInfo.hit = false;
	float T = 100.0f, temp_param;

	for (int i = 0; i < NUM_QUADS; i++)
	{
		if (i != skipIndex)
		{
			tempInfo = intersectQuad(r, quads[i]);
			if (tempInfo.hit == true)
			{
				temp_param = tempInfo.paramT;
				if (temp_param < T)
				{
					T = temp_param;
					tempInfo.objType = QUAD;
					tempInfo.index = i;
					hitInfo = tempInfo;
					hitInfo.reflectance = quad_materials[i].reflectance;
				}
			}
		}
	}
	return hitInfo;
}

HitInfo intersectSphere(Ray r, Sphere s)
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

		if (t1 > 0.0f && t2> 0.0f && t1 < t2)
		{
			hitInfo.hit = true;
			hitInfo.paramT = t1;
		}
	}
	return hitInfo;
}

HitInfo CheckSpheres(Ray r, int skipIndex)
{
	HitInfo hitInfo, tempInfo;
	hitInfo.hit = false;
	float T = 100.0f, temp_param;

	for (int i = 0; i < NUM_SPHERES; i++)
	{
		if (i != skipIndex)
		{
			tempInfo = intersectSphere(r, spheres[i]);
			if (tempInfo.hit == true)
			{
				temp_param = tempInfo.paramT;
				if (temp_param < T)
				{
					T = temp_param;
					tempInfo.objType = SPHERE;
					tempInfo.index = i;
					hitInfo = tempInfo;
					hitInfo.reflectance = sphere_materials[i].reflectance;
				}
			}
		}
	}
	return hitInfo;
}

HitInfo intersectCube(Ray ray, Cube c)
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

HitInfo CheckCubes(Ray r, int skipIndex)
{
	HitInfo hitInfo, tempInfo;
	hitInfo.hit = false;
	float T = 100.0f, temp_param;

	for (int i = 0; i < NUM_CUBES; i++)
	{
		if (i != skipIndex)
		{
			tempInfo = intersectCube(r, cubes[i]);
			if (tempInfo.hit == true)
			{
				temp_param = tempInfo.paramT;
				if (temp_param < T)
				{
					T = temp_param;
					tempInfo.objType = CUBE;
					tempInfo.index = i;
					hitInfo = tempInfo;
					hitInfo.reflectance = cube_materials[i].reflectance;
				}
			}
		}
	}
	return hitInfo;
};

HitInfo getFirstIntersection(Ray r, int skipType, int skipIndex)
{
	HitInfo hiCube, hiSphere, hiQuad, hiFinal;
	hiFinal.hit = false;

	int skip[5] = { -1, -1, -1, -1, -1 };
	skip[skipType] = skipIndex;

	float t1 = 100.0f, t2 = 100.0f;

	hiCube = CheckCubes(r, skip[1]);
	if (hiCube.hit == true)
	{
		t2 = hiCube.paramT;
		if (t2 < t1)
		{
			t1 = t2;
			hiFinal = hiCube;
		}
	}


	hiSphere = CheckSpheres(r, skip[2]);
	if (hiSphere.hit == true)
	{
		t2 = hiSphere.paramT;
		if (t2 < t1)
		{
			t1 = t2;
			hiFinal = hiSphere;
		}
	}

	hiQuad = CheckQuads(r, skip[3]);
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
vec4 phong(Ray r, HitInfo hi, vec3 normal, Material mtl)
{
	vec3 ip = r.getIntersectionPoint(hi.paramT);
	vec3 normalized_light_direction = normalize(vec3(Light1.position) - ip);

	float dist = length(ip - vec3(Light1.position));
	float attn = (1 / (1 + 0.0001f * dist + 0.000003f * dist * dist));

	vec4 ambient = mtl.ambient * 0.2f;

	float tn_dot_ld = max(dot(normal, normalized_light_direction), 0.0f);
	vec4 diffuse = Light1.color * mtl.diffuse * tn_dot_ld * attn;

	vec3 reflection_vector = reflect(normalized_light_direction, normal);
	vec4 specular = mtl.specular * pow(max(dot(reflection_vector, r.dir), 0.0f), mtl.shininess) * attn;

	vec4 color = (ambient + diffuse + specular);
	return  (1 - mtl.reflectance) * color;
}

vec4 TraceShadowRay(Ray r, HitInfo hi, vec3 normal, Material mtl)
{
	vec3 ip = r.getIntersectionPoint(hi.paramT);
	vec3 direction = normalize(vec3(Light1.position) - ip);
	Ray shadowRay = Ray(ip, direction);

	/*if (true)
	{
		HitInfo shadow_info = getFirstIntersection(shadowRay, hi.objType, hi.index);

		if (shadow_info.hit == false)
		{
			return phong(r, hi, normal, mtl);
		}
		else
		{
			return (1 - mtl.reflectance) * (mtl.ambient * 0.3f);
		}
	}
	else*/
		return phong(r, hi, normal, mtl);
}

vec4 calculateColor(Ray r, HitInfo hi)
{
	vec4 color = vec4(0.0f);
	Material mtl;
	vec3 normal;
	switch (hi.objType)
	{
	case CUBE:
		normal = vec3(cubes[hi.index].normal);
		mtl = cube_materials[hi.index];
		break;
	case SPHERE:
		vec3 ip = r.getIntersectionPoint(hi.paramT);
		normal = normalize(ip - vec3(spheres[hi.index].pos));
		mtl = sphere_materials[hi.index];
		break;
	case QUAD:
		normal = vec3(quads[hi.index].normal);
		mtl = quad_materials[hi.index];
		break;
	}

	return TraceShadowRay(r, hi, normal, mtl);
}

Ray getReflectedRay(Ray r, HitInfo hi)
{
	vec3 intersection_point = r.getIntersectionPoint(hi.paramT);
	vec3 normal;

	switch (hi.objType)
	{
	case SPHERE:
		normal = normalize(intersection_point - vec3(spheres[hi.index].pos));
		break;
	case CUBE:
		normal = vec3(cubes[hi.index].normal);
		break;
	case QUAD:
		normal = vec3(quads[hi.index].normal);
		break;
	}
	vec3 reflect_dir = reflect(r.dir, normal);

	return Ray(intersection_point, reflect_dir);
};

vec4 Trace(Ray r)
{
	HitInfo info;
	vec4 color = vec4(0.0f), temp_color;
	float last_reflectance = 1.0f;
	int skipType = NONE, skipIndex = -1;

	for (int i = 0; i <= MAX_BOUNCE; i++)
	{
		info = getFirstIntersection(r, skipType, skipIndex);
		if (info.hit == true)
		{
			temp_color = calculateColor(r, info);
			color = color + last_reflectance * temp_color;

			if (info.reflectance > 0.0f)
			{
				r = getReflectedRay(r, info);
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

vec4 RayTracerStart(int x, int y, vec3 camera, bool bAntiAliasingOn)
{
	vec2 pixel = vec2(x, y);
	Ray ray = initRay(pixel, camera);
	vec4 color;

	if (bAntiAliasingOn)
	{
		Ray r1 = initRay(vec2(pixel.x + 0.25f, pixel.y + 0.25f), camera);
		Ray r2 = initRay(vec2(pixel.x - 0.25f, pixel.y + 0.25f), camera);
		Ray r3 = initRay(vec2(pixel.x + 0.25f, pixel.y - 0.25f), camera);
		Ray r4 = initRay(vec2(pixel.x - 0.25f, pixel.y - 0.25f), camera);

		vec4 c1 = Trace(r1);
		vec4 c2 = Trace(r2);
		vec4 c3 = Trace(r3);
		vec4 c4 = Trace(r4);

		color = (c1 + c2 + c3 + c4) / 4.0f;
	}
	else
		color = Trace(ray);

	return color;
}