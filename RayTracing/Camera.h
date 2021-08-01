#pragma once

#include <gl/glew.h>
#include <gl/GL.h>

#include "glm/mat4x4.hpp"

using namespace glm;

#define CAM_MOVEMENT_SPEED 0.4f
#define CAM_SPIN_SPEED 0.02f
#define CAM_MOUSE_SENSITIVITY 0.01f

class Camera
{
public:
	GLfloat fDir_X;
	GLfloat fDir_Y;
	GLfloat fDir_Z;

	GLfloat fPos_X;
	GLfloat fPos_Y;
	GLfloat fPos_Z;

	GLfloat fUp_X;
	GLfloat fUp_Y;
	GLfloat fUp_Z;

	GLfloat fAngle;

	bool bSpin;

	int iLastX, iLastY;

	Camera();
	mat4 CalculateCameraMatrix();
	void Reset();
	void Spin();
	void MouseMovement(int x, int y);
	void Move(char dir);
};

Camera::Camera()
{
	fDir_X = 0.0f;
	fDir_Y = 0.0f;
	fDir_Z = 0.0f;

	fPos_X = 0.0f;
	fPos_Y = 0.0f;
	fPos_Z = 3.0f;

	fUp_X = 0.0f;
	fUp_Y = 1.0f;
	fUp_Z = 0.0f;

	fAngle = 0.0f;
	bSpin = false;

	iLastX = 400;
	iLastY = 300;
}

mat4 Camera::CalculateCameraMatrix()
{
	vec3 Pos = vec3(fPos_X, fPos_Y, fPos_Z);
	vec3 Dir = vec3(fDir_X, fDir_Y, fDir_Z);
	vec3 Up = vec3(fUp_X, fUp_Y, fUp_Z);

	mat4 cameraMatrix = lookAt(Pos, Dir, Up);
	return cameraMatrix;
}

void Camera::Reset()
{
	fDir_X = 0.0f;
	fDir_Y = 0.0f;
	fDir_Z = 0.0f;

	fPos_X = 0.0f;
	fPos_Y = 0.0f;
	fPos_Z = 3.0f;

	fUp_X = 0.0f;
	fUp_Y = 1.0f;
	fUp_Z = 0.0f;

	fAngle = 0.0f;
	bSpin = false;
}

void Camera::Spin()
{
	if (bSpin)
	{
		if (fAngle < 360.0f)
		{
			fPos_X = 3.0f * sin(fAngle * glm::pi<float>() / 180.0f);
			fPos_Z = 3.0f * cos(fAngle * glm::pi<float>() / 180.0f);
			fAngle += CAM_SPIN_SPEED;
		}
		else
		{
			fAngle = 0.0f;
			bSpin = false;
		}
	}
}

void Camera::Move(char dir)
{
	vec3 Dir = vec3(fDir_X, fDir_Y, fDir_Z);
	vec3 Pos = vec3(fPos_X, fPos_Y, fPos_Z);
	vec3 Mov;
	
	switch (dir)
	{
	case 'U':
		Mov = (Dir - Pos) * CAM_MOVEMENT_SPEED;
		break;
	case 'D':
		Mov = (Pos - Dir) * CAM_MOVEMENT_SPEED;
		break;
	case 'L':
		break;
	case 'R':
		break;
	}

	Pos = Pos + Mov;
	Dir = Dir + Mov;

	fPos_X = Pos[0];
	fPos_Y = Pos[1];
	fPos_Z = Pos[2];

	fDir_X = Dir[0];
	fDir_Y = Dir[1];
	fDir_Z = Dir[2];
}

void Camera::MouseMovement(int x, int y)
{
	int xOffset, yOffset;

	xOffset = x - iLastX;
	yOffset = iLastY - y;

	fDir_X += xOffset * CAM_MOUSE_SENSITIVITY;
	fDir_Y += yOffset * CAM_MOUSE_SENSITIVITY;

	iLastX = x;
	iLastY = y;
}