#include "Scene.h"

/*	Defines the position and colors for all the objects in the scene */

Cube cubes[] =
{
	{
		vec4(-2.0f, -0.5f, 1.5f, 0.0f),
		vec4(-1.5f, 0.5f, 0.8f, 0.0f),
		vec4(1.0f, 0.0f, 1.0f, 0.0f),
	},
	{
		vec4(1.0f, -0.5f, -0.6f, 0.0f),
		vec4(1.3f, -0.2f, -1.0f, 0.0f),
		vec4(1.0f, 0.0f, 1.0f, 0.0f),
	}
};

Material cube_materials[] =
{
	{
		vec4(0.3f, 0.3f, 0.3f, 0.0f),
		vec4(0.0f, 0.4f, 0.6f, 0.0f),
		vec4(0.0f, 0.0f, 0.0f, 0.0f),
		00.0f,
		0.0f
	},
	{
		vec4(0.4f, 0.3f, 0.0f, 0.0f),
		vec4(0.4f, 0.3f, 0.0f, 0.0f),
		vec4(0.0f, 0.0f, 0.0f, 0.0f),
		00.0f,
		0.0f
	}
};

Sphere spheres[] =
{
	{
		vec4(0.5f, -0.3f, -0.2f, 0.2f),
	},
	{
		vec4(0.9f, 0.7f, 0.4f, 0.2f),
	},
	{
		vec4(-0.5f, -0.2f, -0.3f, 0.3f),
	}
};

Material sphere_materials[] =
{
	{
		vec4(0.5f, 0.5f, 0.5f, 0.0f),
		vec4(0.5f, 0.5f, 0.5f, 0.0f),
		vec4(0.0f),
		0.0f,
		0.9f
	},	
	{
		vec4(0.5f, 0.5f, 0.5f, 0.0f),
		vec4(0.5f, 0.5f, 0.5f, 0.0f),
		vec4(0.0f),
		0.0f,
		0.9f
	},
	{
		vec4(0.0f, 1.0f, 0.3f, 0.0f),
		vec4(0.0f, 1.0f, 0.3f, 0.0f),
		vec4(0.0f),
		0.0f,
		0.3f
	}
};


Quad quads[] =
{
	// Bottom
	{
		vec4(-3.0f, -0.5f, 5.2f, 0.0f),
		vec4(-3.0f, -0.5f, -1.8f, 0.0f),
		vec4(3.0f, -0.5f, -1.8f, 0.0f),
		vec4(3.0f, -0.5f, 5.2f, 0.0f),
		vec4(0.0f, 1.0f, 0.0f, 0.0f)
	}
};

Material quad_materials[] = 
{
	{
		vec4(0.0f, 0.0f, 0.0f, 0.0f),
		vec4(1.0f, 1.0f, 1.0f, 0.0f),
		vec4(0.0f, 0.0f, 0.0f, 0.0f),
		50.0f,
		0.5f
	}
};


Light Light1 = { 
	vec4(1.5f, 1.3f, -0.3f, 0.4f),
	vec4(1.0f, 1.0f, 1.0f, 0.0f),
	1000.0f 
};
