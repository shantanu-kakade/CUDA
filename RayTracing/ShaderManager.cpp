#include "ShaderManager.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>

using namespace std;

string cleanString(string str)
{
	string cleaned_string;

	string n = "\\n";
	int pos = str.find(n);
	if (pos != string::npos)
	{
		cleaned_string += "\n";
	}
	else
		cleaned_string += str;

	return cleaned_string;
}

FILE* ShaderManager::m_LogFile = NULL;

int ShaderManager::CreateShader(const GLchar* shaderFilePath, GLenum shaderType)
{
	fopen_s(&m_LogFile, "Shader Log.txt", "a");

	int shaderObject;

	string content, line;
	ifstream shaderFile(shaderFilePath);
	if (shaderFile.fail())
	{
		fprintf(m_LogFile, "Cannot open file %s", shaderFilePath);
		return -1;
	}

	while (getline(shaderFile, line))
		content.append(cleanString(line));
	
	shaderFile.close();

	shaderObject = CompileShader(&content[0], shaderFilePath, shaderType);

	fclose(m_LogFile);
	return shaderObject;
}

int ShaderManager::CompileShader(const GLchar* shaderCode, const GLchar* shaderFilePath, GLenum shaderType)
{
	// Compile shader
	int shaderObject = glCreateShader(shaderType);

	glShaderSource(shaderObject, 1, (const GLchar**)&shaderCode, NULL);
	glCompileShader(shaderObject);

	GLint iShaderCompileStatus = 0, iInfoLogLength = 0;
	char *szInfoLog = NULL;

	glGetShaderiv(shaderObject, GL_COMPILE_STATUS, &iShaderCompileStatus);
	if (iShaderCompileStatus == GL_FALSE)
	{
		glGetShaderiv(shaderObject, GL_INFO_LOG_LENGTH, &iInfoLogLength);
		if (iInfoLogLength > 0)
		{
			szInfoLog = (char*)malloc(iInfoLogLength);
			if (szInfoLog != NULL)
			{
				GLsizei written;
				glGetShaderInfoLog(shaderObject, iInfoLogLength, &written, szInfoLog);
				fprintf(m_LogFile, "\n%s compile log:\n %s\n\n", shaderFilePath, szInfoLog);
				free(szInfoLog);

				return -1;
			}
		}		
	}
	else
		fprintf(m_LogFile, "%s compiled successfully\n\n", shaderFilePath);

	return shaderObject;
}

int ShaderManager::CreateShaderProgram(int shaderCount ...)
{
	// Initialize log file
	fopen_s(&m_LogFile, "Shader Log.txt", "a");

	int ShaderProgramObject, ShaderObject;

	ShaderProgramObject = glCreateProgram();

	va_list args;
	va_start(args, shaderCount);

	for (int i = 0; i < shaderCount; i++)
	{
		glAttachShader(ShaderProgramObject, va_arg(args, int));
	}

	return LinkShader(ShaderProgramObject);
}

int ShaderManager::LinkShader(int ShaderProgramObject)
{
	glLinkProgram(ShaderProgramObject);

	GLint iShaderLinkStatus = 0, iInfoLogLength = 0;
	char *szInfoLog = NULL;

	glGetProgramiv(ShaderProgramObject, GL_LINK_STATUS, &iShaderLinkStatus);
	if (iShaderLinkStatus == GL_FALSE)
	{
		glGetProgramiv(ShaderProgramObject, GL_INFO_LOG_LENGTH, &iInfoLogLength);
		if (iInfoLogLength > 0)
		{
			szInfoLog = (char*)malloc(iInfoLogLength);
			if (szInfoLog != NULL)
			{
				GLsizei written;
				glGetProgramInfoLog(ShaderProgramObject, iInfoLogLength, &written, szInfoLog);
				fprintf(m_LogFile, "Linking log:\n %s\n", szInfoLog);
				free(szInfoLog);

				return -1;
			}
		}
	}
	else
		fprintf(m_LogFile, "Linked successfully\n\n");

	fclose(m_LogFile);

	return ShaderProgramObject;
}