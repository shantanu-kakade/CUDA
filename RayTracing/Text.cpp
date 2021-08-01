#include <map>
#include <string>
#include <sstream>
#include <iomanip>
#include <gl/glew.h>
#include <gl/GL.h>

#include "glm/vec3.hpp"
#include "glm/vec4.hpp"
#include "glm/mat4x4.hpp"
#include "glm/ext/matrix_transform.hpp"	// For glm::identity
#include "glm/ext.hpp"					// For value_ptr

#include "ShaderManager.h"
#include "ft2build.h"
#include FT_FREETYPE_H

#pragma comment(lib, "freetype.lib")

using namespace glm;

GLuint gVao_FPS;
GLuint gVbo_FPS_position;
GLuint gVbo_FPS_texture;

GLuint gVao_CudaLabel;
GLuint gVbo_CudaLabel_position;
GLuint gVbo_CudaLabel_texture;

GLuint gVS_Text, gFS_Text;
GLuint gShaderProgram_Text;

GLuint gMVPUniform_TextShader;
GLuint gTexture_sampler_uniform;

struct CharacterGlyph
{
	GLuint textureID;
	vec2 size;
	vec2 bearing;
	GLuint advance;
};

std::map<char, CharacterGlyph> LoadedGlyphs;

int giLastFPSDisplay = 0;
float lastFPS;

int loadChar(std::string str, FT_Face face, FILE *logFile)
{
	for (int i = 0; i < str.length(); i++)
	{
		char c = str[i];
		if (FT_Load_Char(face, c, FT_LOAD_RENDER))
		{
			fprintf(logFile, "Could not load character %c \n", c);
			return -1;
		}

		GLuint textureID;
		glGenTextures(1, &textureID);

		glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
		glBindTexture(GL_TEXTURE_2D, textureID);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

		glTexImage2D(GL_TEXTURE_2D,
			0,
			GL_RED,
			face->glyph->bitmap.width,
			face->glyph->bitmap.rows,
			0,
			GL_RED,
			GL_UNSIGNED_BYTE,
			face->glyph->bitmap.buffer);

		glGenerateMipmap(GL_TEXTURE_2D);

		CharacterGlyph cg = { textureID,
			vec2(face->glyph->bitmap.width, face->glyph->bitmap.rows),
			vec2(face->glyph->bitmap_left, face->glyph->bitmap_top),
			face->glyph->advance.x };

		LoadedGlyphs.insert(std::pair<char, CharacterGlyph>(c, cg));
	}
	return 0;
}

void InitText(FILE *logFile)
{
	void uninitialize();

	FT_Library ft;
	if (FT_Init_FreeType(&ft))
	{
		fprintf(logFile, "Error initializing freetype \n");
		return;
	}

	FT_Face face;
	if (FT_New_Face(ft, "Fonts/arial.ttf", 0, &face))
	{
		fprintf(logFile, "Could not load face \n");
		return;
	}

	FT_Set_Pixel_Sizes(face, 0, 48);

	// Initialize glyphs of all required characters
	loadChar("FPS:.", face, logFile);
	loadChar("CUDA-On", face, logFile);
	loadChar("CUDA-Off", face, logFile);

	std::string zero = std::to_string(0); // A separate call is needed for zero because converting 01234 to string doesn't work as expected
	loadChar(zero, face, logFile);
	std::string str0 = std::to_string(1234);
	loadChar(str0, face, logFile);
	std::string str5 = std::to_string(56789);
	loadChar(str5, face, logFile);

	FT_Done_Face(face);
	FT_Done_FreeType(ft);

	const GLfloat TexCoords[] = {
		0.0f, 0.0f,
		0.0f, 1.0f,
		1.0f, 1.0f,
		1.0f, 0.0f
	};

	// Vertex Shader
	gVS_Text = ShaderManager::CreateShader("Shader/Vertex Shader Text.txt", GL_VERTEX_SHADER);
	if (gVS_Text == -1)
	{
		uninitialize();
		exit(0);
	}

	// Fragment Shader
	gFS_Text = ShaderManager::CreateShader("Shader/Fragment Shader Text.txt", GL_FRAGMENT_SHADER);
	if (gFS_Text == -1)
	{
		uninitialize();
		exit(0);
	}

	gShaderProgram_Text = ShaderManager::CreateShaderProgram(2, gVS_Text, gFS_Text);
	if (gShaderProgram_Text == -1)
	{
		uninitialize();
		exit(0);
	}

	gMVPUniform_TextShader = glGetUniformLocation(gShaderProgram_Text, "u_mvp_matrix");
	gTexture_sampler_uniform = glGetUniformLocation(gShaderProgram_Text, "u_texture0_sampler");

	// Cuda label position
	const GLfloat squareVertices[] = {
		-0.07f, 0.07f,0.0f,
		-0.07f,-0.07f, 0.0f,
		0.07f,-0.07f,0.0f,
		0.07f, 0.07f, 0.0f
	};

	// Generate Vao FPS
	glGenVertexArrays(1, &gVao_FPS);
	glBindVertexArray(gVao_FPS);

	// FPS position
	glGenBuffers(1, &gVbo_FPS_position);
	glBindBuffer(GL_ARRAY_BUFFER, gVbo_FPS_position);

	glBufferData(GL_ARRAY_BUFFER, sizeof(squareVertices), squareVertices, GL_STATIC_DRAW);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// FPS texture
	glGenBuffers(1, &gVbo_FPS_texture);
	glBindBuffer(GL_ARRAY_BUFFER, gVbo_FPS_texture);

	glBufferData(GL_ARRAY_BUFFER, sizeof(TexCoords), TexCoords, GL_STATIC_DRAW);
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(1);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindVertexArray(0);

	// Generate Vao Cuda Label
	glGenVertexArrays(1, &gVao_CudaLabel);
	glBindVertexArray(gVao_CudaLabel);

	// Cuda Label position
	glGenBuffers(1, &gVbo_CudaLabel_position);
	glBindBuffer(GL_ARRAY_BUFFER, gVbo_CudaLabel_position);

	glBufferData(GL_ARRAY_BUFFER, sizeof(squareVertices), squareVertices, GL_STATIC_DRAW);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// Cuda Label texture
	glGenBuffers(1, &gVbo_CudaLabel_texture);
	glBindBuffer(GL_ARRAY_BUFFER, gVbo_CudaLabel_texture);

	glBufferData(GL_ARRAY_BUFFER, sizeof(TexCoords), TexCoords, GL_STATIC_DRAW);
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(1);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindVertexArray(0);
}

void displayString(std::string str, mat4 perspectiveProjectionMatrix, vec3 position)
{
	mat4 modelViewMatrix = mat4(1.0f);
	mat4 modelViewProjectionMatrix = mat4(1.0f);
	
	for (int i = 0; i < str.length(); i++)
	{
		char c = str[i];

		CharacterGlyph cg = LoadedGlyphs[c];
		
		position[0] += 0.15f; // Move x co-ordinate of each letter to right
		modelViewMatrix = translate(mat4(1.0f), vec3(position[0], position[1], position[2]));
		modelViewProjectionMatrix = perspectiveProjectionMatrix * modelViewMatrix;
		glUniformMatrix4fv(gMVPUniform_TextShader, 1, GL_FALSE, value_ptr(modelViewProjectionMatrix));

		glBindTexture(GL_TEXTURE_2D, cg.textureID);
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
	}
}

void displayText(mat4 perspectiveProjectionMatrix, float FPS, bool bCuda)
{
	glUseProgram(gShaderProgram_Text);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	mat4 modelViewMatrix = mat4(1.0f);
	mat4 modelViewProjectionMatrix = mat4(1.0f);

	modelViewMatrix = translate(modelViewMatrix, vec3(3.0f, 2.0f, -6.0f));

	modelViewProjectionMatrix = perspectiveProjectionMatrix * modelViewMatrix;
	glUniformMatrix4fv(gMVPUniform_TextShader, 1, GL_FALSE, value_ptr(modelViewProjectionMatrix));

	glBindVertexArray(gVao_FPS);

	glActiveTexture(GL_TEXTURE0);
	glUniform1i(gTexture_sampler_uniform, 0);
	
	displayString("CUDA", perspectiveProjectionMatrix, vec3(1.7f, 1.0f, -3.0f));

	if (bCuda)
		displayString("On", perspectiveProjectionMatrix, vec3(2.4f, 1.0f, -3.0f));
	else
		displayString("Off", perspectiveProjectionMatrix, vec3(2.4f, 1.0f, -3.0f));

	displayString("FPS", perspectiveProjectionMatrix, vec3(1.7f, 1.3f, -3.0f));

	// Display new FPS value after an interval of 1 sec, to avoid flickering
	DWORD currentTick = GetTickCount();
	if ((currentTick - giLastFPSDisplay) > 1000)
	{
		giLastFPSDisplay = currentTick;
		lastFPS = FPS;
		displayString(std::to_string((int)FPS), perspectiveProjectionMatrix, vec3(2.3f, 1.3f, -3.0f));
	}
	else
		displayString(std::to_string((int)lastFPS), perspectiveProjectionMatrix, vec3(2.3f, 1.3f, -3.0f));

	glBindVertexArray(0);

	glDisable(GL_BLEND);

	glUseProgram(0);
}