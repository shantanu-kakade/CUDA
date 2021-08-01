#pragma once

#include <stdlib.h>
#include <stdio.h>
#include <Windows.h>

#include <GL/glew.h>
#include <gl/GL.h>


#pragma comment(lib, "opengl32.lib")

class ShaderManager
{
public:
	static FILE* m_LogFile;

	static int CreateShaderProgram(int shaderCount ...);

	static int CreateShader(const GLchar* shaderFilePath, GLenum shaderType);
	static int CompileShader(const GLchar* shaderCode, const GLchar* shaderFilePath, GLenum shaderType);
	static int LinkShader(int Shader);
};