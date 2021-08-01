#include <Windows.h>
#include <stdio.h>
#include <gl/glew.h>
#include <gl/GL.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "glm/mat4x4.hpp"
#include "glm/ext/matrix_transform.hpp"	// For glm::identity
#include "glm/ext.hpp"					// For glm::perspective

#include "ShaderManager.h"
#include "Camera.h"
#include "RayTracer.h"

#pragma comment(lib, "glew32.lib")
#pragma comment(lib, "opengl32.lib")

using namespace glm;

#define WIN_WIDTH 800
#define WIN_HEIGHT 600

LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);

FILE* gpFile = NULL;

bool gbFullscreen = false;
bool gbActiveWindow = false;
bool gbEscPressed = false;

HWND gHwnd = NULL;
HDC gHdc = NULL;
HGLRC gHrc = NULL;

WINDOWPLACEMENT wpPrev;
DWORD dwStyle;

extern class RayTracer RT;

enum
{
	ATTRIBUTE_VERTEX = 0,
	ATTRIBUTE_COLOR,
	ATTRIBUTE_NORMAL,
	ATTRIBUTE_TEXTURE0
};

bool gbAnimate = true;
bool gbCudaOn = true;
bool gbShadowOn = false;
bool gbAntiAliasingOn = false;

GLint gVertexShaderObject;
GLint gFragmentShaderObject;
GLint gShaderProgramObject;

GLuint gVao;
GLuint gVbo_position;
GLuint gVbo_texture;

GLuint gTexture;

GLuint gMVPUniform;

mat4 gPerspectiveProjectionMatrix;

GLfloat gfAngle;

Camera Cam1;

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpscCmdLine, int nCmdShow)
{
	void initialize();
	void display(float FPS);
	void update();
	void uninitialize();
	void displayText(mat4 perspectiveProjectionMatrix, float FPS, bool bCuda);

	WNDCLASSEX wndclass;
	HWND hwnd;
	MSG msg;
	TCHAR szAppName[] = TEXT("Ray Tracing CUDA");
	bool bDone = false;

	if (fopen_s(&gpFile, "Log.txt", "w") != 0)
	{
		MessageBox(NULL, TEXT("Cannot open log file. Exiting......"), TEXT("Error"), MB_OK);
		exit(0);
	}
	else
		fprintf(gpFile, "Log file successfully opened. \n");

	wndclass.cbSize = sizeof(WNDCLASSEX);
	wndclass.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC;
	wndclass.cbClsExtra = 0;
	wndclass.cbWndExtra = 0;
	wndclass.lpfnWndProc = WndProc;
	wndclass.hInstance = hInstance;
	wndclass.hIcon = LoadIcon(NULL, IDI_APPLICATION);
	wndclass.hIconSm = LoadIcon(NULL, IDI_APPLICATION);
	wndclass.hCursor = LoadCursor(NULL, IDC_ARROW);
	wndclass.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH);
	wndclass.lpszClassName = szAppName;
	wndclass.lpszMenuName = NULL;

	RegisterClassEx(&wndclass);

	hwnd = CreateWindowEx(WS_EX_APPWINDOW,
		szAppName,
		TEXT("Ray Tracing CUDA"),
		WS_OVERLAPPEDWINDOW,
		GetSystemMetrics(SM_CXSCREEN) / 2 - 400,
		GetSystemMetrics(SM_CYSCREEN) / 2 - 300,
		WIN_WIDTH,
		WIN_HEIGHT,
		NULL,
		NULL,
		hInstance,
		NULL);

	gHwnd = hwnd;

	initialize();

	ShowWindow(hwnd, SW_SHOW);
	SetForegroundWindow(hwnd);
	SetFocus(hwnd);

	DWORD previousTime = GetTickCount();
	while (bDone == false)
	{
		if (PeekMessage(&msg, hwnd, 0, 0, PM_REMOVE))
		{
			if (msg.message == WM_QUIT)
				bDone = true;
			else
			{
				TranslateMessage(&msg);
				DispatchMessage(&msg);
			}
		}
		else
		{
			if (gbActiveWindow)
			{
				/* FPS is calculated as the amount of time display() is called in one second 
				   Divide 1000 by the time elapsed since previous display() call in millisecond to get FPS value
				*/
				DWORD currentTime = GetTickCount();
				int elapsedTime = currentTime - previousTime;
				previousTime = currentTime;
				float FPS = 0.0f;
				if (elapsedTime != 0)
					FPS = 1000.0f / elapsedTime;

				display(FPS);
				update();

				if (gbEscPressed)
					bDone = true;
			}
		}
	}

	uninitialize();
	return ((int)msg.wParam);
}

LRESULT CALLBACK WndProc(HWND hwnd, UINT iMsg, WPARAM wParam, LPARAM lParam)
{
	void resize(int, int);
	void ToggleFullscreen();

	switch (iMsg)
	{
	case WM_ACTIVATE:
		if (HIWORD(wParam) == 0)
			gbActiveWindow = true;
		else
			gbActiveWindow = false;
		break;
	case WM_SIZE:
		resize(LOWORD(lParam), HIWORD(lParam));
		break;
	case WM_KEYDOWN:
	{
		switch (wParam)
		{
		case VK_ESCAPE:
			gbEscPressed = true;
			break;
		case 0x46:	// F
			if (gbFullscreen == false)
			{
				ToggleFullscreen();
				gbFullscreen = true;
			}
			else
			{
				ToggleFullscreen();
				gbFullscreen = false;
			}
			break;
		case VK_SPACE:
			gbAnimate = !gbAnimate;
			break;
		case 0x32:	// 2
			gbShadowOn = !gbShadowOn;
			break;
		case 0x33:	// 3
			gbAntiAliasingOn = !gbAntiAliasingOn;
			break;
		case 0x43:	// C
			gbCudaOn = !gbCudaOn;
			break;
		case 0x52:	// R
			break;
			// Camera controls
		case 0x57:	// W
			Cam1.Move('U');
			break;
		case 0x53:	// S
			Cam1.Move('D');
			break;
		case 0x41:	// A
			Cam1.fPos_X -= CAM_MOVEMENT_SPEED;
			Cam1.fDir_X -= CAM_MOVEMENT_SPEED;
			break;
		case 0x44:	// D
			Cam1.fPos_X += CAM_MOVEMENT_SPEED;
			Cam1.fDir_X += CAM_MOVEMENT_SPEED;
			break;
		case VK_UP:
			Cam1.fPos_Y += CAM_MOVEMENT_SPEED;
			Cam1.fDir_Y += CAM_MOVEMENT_SPEED;
			break;
		case VK_DOWN:
			Cam1.fPos_Y -= CAM_MOVEMENT_SPEED;
			Cam1.fDir_Y -= CAM_MOVEMENT_SPEED;
			break;
		default:
			break;
		}
		break;
	}
	case WM_DESTROY:
		PostQuitMessage(0);
		break;
	}

	return (DefWindowProc(hwnd, iMsg, wParam, lParam));
}

void ToggleFullscreen()
{
	MONITORINFO mi;

	if (gbFullscreen == false)
	{
		dwStyle = GetWindowLong(gHwnd, GWL_STYLE);
		if (dwStyle & WS_OVERLAPPEDWINDOW)
		{
			mi = { sizeof(MONITORINFO) };
			if (GetWindowPlacement(gHwnd, &wpPrev) && GetMonitorInfo(MonitorFromWindow(gHwnd, MONITORINFOF_PRIMARY), &mi))
			{
				SetWindowLong(gHwnd, GWL_STYLE, dwStyle & ~WS_OVERLAPPEDWINDOW);
				SetWindowPos(gHwnd, HWND_TOP, mi.rcMonitor.left, mi.rcMonitor.top, mi.rcMonitor.right - mi.rcMonitor.left, mi.rcMonitor.bottom - mi.rcMonitor.top, SWP_NOZORDER | SWP_FRAMECHANGED);
			}
		}
		ShowCursor(FALSE);
	}
	else
	{
		SetWindowLong(gHwnd, GWL_STYLE, dwStyle | WS_OVERLAPPEDWINDOW);
		SetWindowPlacement(gHwnd, &wpPrev);
		SetWindowPos(gHwnd, HWND_TOP, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOOWNERZORDER | SWP_NOZORDER | SWP_FRAMECHANGED);
		ShowCursor(TRUE);
	}
}

void initialize()
{
	void uninitialize();
	void resize(int, int);
	void InitText(FILE * logFile);

	PIXELFORMATDESCRIPTOR pfd;
	int iPixelFormatIndex;

	ZeroMemory(&pfd, sizeof(PIXELFORMATDESCRIPTOR));

	pfd.nSize = sizeof(PIXELFORMATDESCRIPTOR);
	pfd.nVersion = 1;
	pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
	pfd.iPixelType = PFD_TYPE_RGBA;
	pfd.cColorBits = 32;
	pfd.cRedBits = 8;
	pfd.cGreenBits = 8;
	pfd.cBlueBits = 8;
	pfd.cAlphaBits = 8;
	pfd.cDepthBits = 32;

	gHdc = GetDC(gHwnd);

	iPixelFormatIndex = ChoosePixelFormat(gHdc, &pfd);

	if (iPixelFormatIndex == 0)
	{
		ReleaseDC(gHwnd, gHdc);
		gHdc = NULL;
	}

	if (SetPixelFormat(gHdc, iPixelFormatIndex, &pfd) == FALSE)
	{
		ReleaseDC(gHwnd, gHdc);
		gHdc = NULL;
	}

	gHrc = wglCreateContext(gHdc);
	if (gHrc == NULL)
	{
		ReleaseDC(gHwnd, gHdc);
		gHdc = NULL;
	}

	if (wglMakeCurrent(gHdc, gHrc) == FALSE)
	{
		wglDeleteContext(gHrc);
		gHrc = NULL;

		ReleaseDC(gHwnd, gHdc);
		gHdc = NULL;
	}

	GLenum glew_error = glewInit();
	if (glew_error != GLEW_OK)
	{
		wglDeleteContext(gHrc);
		gHrc = NULL;

		ReleaseDC(gHwnd, gHdc);
		gHdc = NULL;
	}

	// Vertex Shader
	gVertexShaderObject = ShaderManager::CreateShader("Shader/Vertex Shader.txt", GL_VERTEX_SHADER);
	if (gVertexShaderObject == -1)
	{
		uninitialize();
		exit(0);
	}

	// Fragment Shader
	gFragmentShaderObject = ShaderManager::CreateShader("Shader/Fragment Shader.txt", GL_FRAGMENT_SHADER);
	if (gFragmentShaderObject == -1)
	{
		uninitialize();
		exit(0);
	}

	gShaderProgramObject = ShaderManager::CreateShaderProgram(2, gVertexShaderObject, gFragmentShaderObject);
	if (gShaderProgramObject == -1)
	{
		uninitialize();
		exit(0);
	}

	gMVPUniform = glGetUniformLocation(gShaderProgramObject, "u_mvp_matrix");

	// Position of the texQuad on which the texture will be drawn
	const GLfloat pos[] =
	{
		-2.98f, 1.69f, 0.0f,
		-2.98f, -1.69f, 0.0f,
		2.98f, -1.69f, 0.0f,
		2.98f, 1.69f, 0.0f
	};

	const GLfloat tex[] =
	{
		0.0f, float(TEX_HEIGHT) / float(TEX_WIDTH),
		0.0f, 0.0f,
		1.0f, 0.0f,
		1.0f, float(TEX_HEIGHT) / float(TEX_WIDTH)
	};

	glGenVertexArrays(1, &gVao);
	glBindVertexArray(gVao);

	// Texture Position
	glGenBuffers(1, &gVbo_position);
	glBindBuffer(GL_ARRAY_BUFFER, gVbo_position);

	glBufferData(GL_ARRAY_BUFFER, sizeof(pos), pos, GL_STATIC_DRAW);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(0);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// Texture vbo
	glGenBuffers(1, &gVbo_texture);
	glBindBuffer(GL_ARRAY_BUFFER, gVbo_texture);

	glBufferData(GL_ARRAY_BUFFER, sizeof(tex), tex, GL_STATIC_DRAW);
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(1);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);

	glClearDepth(1.0f);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glShadeModel(GL_SMOOTH);
	glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
	//glEnable(GL_CULL_FACE);

	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

	gPerspectiveProjectionMatrix = mat4(1.0f);

	// Initialize FreeType library 
	InitText(gpFile);

	// Initialize RayTracer class and the CUDA memory required
	RT.init();

	resize(WIN_WIDTH, WIN_HEIGHT);
}

void display(float FPS)
{
	void displayText(mat4 perspectiveProjectionMatrix, float FPS, bool bCuda);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glUseProgram(gShaderProgramObject);

	mat4 modelViewMatrix = mat4(1.0f);
	mat4 modelViewProjectionMatrix = mat4(1.0f);

	modelViewMatrix = translate(modelViewMatrix, vec3(0.0f, 0.0f, -3.0f));

	modelViewProjectionMatrix = gPerspectiveProjectionMatrix * modelViewMatrix;

	glUniformMatrix4fv(gMVPUniform, 1, GL_FALSE, value_ptr(modelViewProjectionMatrix));

	glUseProgram(gShaderProgramObject);

	glBindVertexArray(gVao);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, gTexture);
	glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glUseProgram(0);

	displayText(gPerspectiveProjectionMatrix, FPS, gbCudaOn);
	SwapBuffers(gHdc);
}

void update()
{
	if (gbAnimate)
		RT.render(gTexture, gbCudaOn, gbShadowOn, gbAntiAliasingOn);
}

void resize(int width, int height)
{
	if (height == 0)
		height = 1;

	glViewport(0, 0, (GLsizei)width, (GLsizei)height);

	gPerspectiveProjectionMatrix = perspective(45.0f, (GLfloat)width / (GLfloat)height, 0.1f, 200.0f);
}

void uninitialize()
{
	if (gbFullscreen == true)
	{
		dwStyle = GetWindowLong(gHwnd, GWL_STYLE);
		SetWindowLong(gHwnd, GWL_STYLE, dwStyle | WS_OVERLAPPEDWINDOW);
		SetWindowPlacement(gHwnd, &wpPrev);
		SetWindowPos(gHwnd, HWND_TOP, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOOWNERZORDER | SWP_NOZORDER | SWP_FRAMECHANGED);
		ShowCursor(true);
	}

	if (gVao)
	{
		glDeleteVertexArrays(1, &gVao);
		gVao = 0;
	}

	if (gVbo_position)
	{
		glDeleteBuffers(1, &gVbo_position);
		gVbo_position = 0;
	}
	if (gVbo_texture)
	{
		glDeleteBuffers(1, &gVbo_texture);
		gVbo_texture = 0;
	}

	glDetachShader(gShaderProgramObject, gFragmentShaderObject);
	glDetachShader(gShaderProgramObject, gVertexShaderObject);

	glDeleteShader(gVertexShaderObject);
	gVertexShaderObject = 0;

	glDeleteShader(gFragmentShaderObject);
	gFragmentShaderObject = 0;

	glDeleteProgram(gShaderProgramObject);
	gShaderProgramObject = 0;

	glUseProgram(0);

	wglMakeCurrent(NULL, NULL);

	wglDeleteContext(gHrc);
	gHrc = NULL;

	ReleaseDC(gHwnd, gHdc);
	gHdc = NULL;

	DestroyWindow(gHwnd);

	if (gpFile)
	{
		fprintf(gpFile, "Log file successfully closed");
		fclose(gpFile);
		gpFile = NULL;
	}
}

