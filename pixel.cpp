/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */
 

// OpenGL Graphics includes
#include <GL/glew.h>
#if defined(__APPLE__) || defined(__MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif

// CUDA utilities and system includes
#include <cutil_inline.h>    // includes cuda.h and cuda_runtime_api.h    // includes cuda_runtime_api.h
#include <cutil_gl_inline.h> // includes cuda_gl_interop.h// includes cuda_gl_interop.h

#include <shrUtils.h>

#define MAX_EPSILON_ERROR 5.0f
#define REFRESH_DELAY	  10 //ms

#define MIN_RUNTIME_VERSION 1000
#define MIN_COMPUTE_VERSION 0x10


const char *image_filename = "lenaRGB.ppm";
//const char *vector_filename = "lenaRGB1.ppm";
//int iterations = 0;
//int filter_radius = 2;
int nthreads = 16;


unsigned int width, height;
unsigned int scale = 16;
unsigned int * h_img = NULL;
unsigned int * h_result = NULL;
unsigned int * d_img = NULL;
unsigned int * d_temp = NULL;
float2       * d_point = NULL;

GLuint pbo;     // OpenGL pixel buffer object
struct cudaGraphicsResource *cuda_pbo_resource; // handles OpenGL-CUDA exchange
GLuint texid;   // texture
GLuint shader;

GLuint pboresult; // Output result
struct cudaGraphicsResource *cuda_pboresult_resource;
GLuint texidresult;

unsigned int timer = 0;

// Auto-Verification Code
const int frameCheckNumber = 4;
int fpsCount = 0;        // FPS count for averaging
int fpsLimit = 1;        // FPS limit for sampling
int g_Index = 0;
unsigned int frameCount = 0;
unsigned int g_TotalErrors = 0;


//#define GL_TEXTURE_TYPE GL_TEXTURE_RECTANGLE_ARB
#define GL_TEXTURE_TYPE GL_TEXTURE_2D

extern "C" void loadImageData(int argc, char **argv);


// These are CUDA functions to handle allocation and launching the kernels
extern "C" void initTexture(int width, int height, void *pImage, void *pResult);
extern "C" void freeTextures();

extern "C" double connectivityDetection(uint *d_temp, unsigned int *d_dest, unsigned int *d_dest2, float2 *d_point, int width, int height, int scale, int nthreads);


// display results using OpenGL
void display()
{
    cutilCheckError(cutStartTimer(timer));  

    // execute filter, writing results to pbo
    unsigned int *d_result;
	unsigned int *d_pboresult;
	double elapsed_time;
    //DEPRECATED: cutilSafeCall( cudaGLMapBufferObject((void**)&d_result, pbo) );
    cutilSafeCall(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
	cutilSafeCall(cudaGraphicsMapResources(1, &cuda_pboresult_resource, 0));
    size_t num_bytes; 
	size_t num_bytes_result;
    cutilSafeCall(cudaGraphicsResourceGetMappedPointer((void **)&d_result, &num_bytes,  
						       cuda_pbo_resource));
	cutilSafeCall(cudaGraphicsResourceGetMappedPointer((void **)&d_pboresult, &num_bytes_result,
							   cuda_pboresult_resource));
	elapsed_time = connectivityDetection(d_temp, d_result, d_pboresult, d_point, width, height, scale, nthreads);
	//printf("%f\n", elapsed_time);
    // DEPRECATED: cutilSafeCall(cudaGLUnmapBufferObject(pbo));
    cutilSafeCall(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));
	cutilSafeCall(cudaGraphicsUnmapResources(1, &cuda_pboresult_resource, 0));

    
    // Common display code path
    {
        glClear(GL_COLOR_BUFFER_BIT);
	
        // load texture from pbo
        glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pboresult);
        glBindTexture(GL_TEXTURE_2D, texidresult);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width*scale, height*scale, GL_RGBA, GL_UNSIGNED_BYTE, 0);
        glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

        // fragment program is required to display floating point texture
        glBindProgramARB(GL_FRAGMENT_PROGRAM_ARB, shader);
        glEnable(GL_FRAGMENT_PROGRAM_ARB);
        glDisable(GL_DEPTH_TEST);

        glBegin(GL_QUADS);
        if (GL_TEXTURE_TYPE == GL_TEXTURE_2D) {
            glTexCoord2f(0.0f, 0.0f);          
            glVertex2f(0.0f, 0.0f);
            glTexCoord2f(1.0f, 0.0f);          
            glVertex2f(1.0f, 0.0f);
            glTexCoord2f(1.0f, 1.0f);          
            glVertex2f(1.0f, 1.0f);
            glTexCoord2f(0.0f, 1.0f);          
            glVertex2f(0.0f, 1.0f);
        } else {
            glTexCoord2f(0.0f, 0.0f); 
            glVertex2f(0.0f, 0.0f);
            glTexCoord2f((float)width*scale, 0.0f); 
            glVertex2f(1.0f, 0.0f);
            glTexCoord2f((float)width*scale, (float)height*scale); 
            glVertex2f(1.0f, 1.0f);
            glTexCoord2f(0.0f, (float)height*scale); glVertex2f(0.0f, 1.0f);
        }
        glEnd();
        glBindTexture(GL_TEXTURE_TYPE, 0);
        glDisable(GL_FRAGMENT_PROGRAM_ARB);
    }

	glutSwapBuffers();
    glutReportErrors();

    cutilCheckError(cutStopTimer(timer));  

}


void keyboard(unsigned char key, int /*x*/, int /*y*/)
{
	switch(key) {
		case 27:
		exit(0);
		break;
		case '=':
		case '+':
		case '-':
		case ']':
		case '[':
		default:
		break;
	}
}


void timerEvent(int value)
{
    glutPostRedisplay();
	glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
}

void reshape(int x, int y)
{
    glViewport(0, 0, x, y);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0); 
}

void initCuda()
{
    // allocate device memory
    cutilSafeCall( cudaMalloc( (void**) &d_img,  (width * height * sizeof(unsigned int)) ));
    cutilSafeCall( cudaMalloc( (void**) &d_temp, (width * height * sizeof(unsigned int)) ));
	cutilSafeCall( cudaMalloc( (void**) &d_point, (width * height * sizeof(float2) * 8) ));

    // Refer to pixel_kernel.cu for implementation
    initTexture(width, height, h_img, h_result); 

    cutilCheckError( cutCreateTimer( &timer));
}

void cleanup()
{
    cutilCheckError( cutDeleteTimer( timer));
    if(h_img)cutFree(h_img);
	if(h_result) free(h_result);
    cutilSafeCall(cudaFree(d_img));
    cutilSafeCall(cudaFree(d_temp));

    // Refer to pixel_kernel.cu for implementation
    freeTextures();

    //DEPRECATED: cutilSafeCall(cudaGLUnregisterBufferObject(pbo));    
    cudaGraphicsUnregisterResource(cuda_pbo_resource);
	cudaGraphicsUnregisterResource(cuda_pboresult_resource);

    glDeleteBuffersARB(1, &pbo);
	glDeleteBuffersARB(1, &pboresult);
    glDeleteTextures(1, &texid);
	glDeleteTextures(1, &texidresult);
    glDeleteProgramsARB(1, &shader);

 }

// shader for displaying floating-point texture
static const char *shader_code = 
"!!ARBfp1.0\n"
"TEX result.color, fragment.texcoord, texture[0], 2D; \n"
"END";

GLuint compileASMShader(GLenum program_type, const char *code)
{
    GLuint program_id;
    glGenProgramsARB(1, &program_id);
    glBindProgramARB(program_type, program_id);
    glProgramStringARB(program_type, GL_PROGRAM_FORMAT_ASCII_ARB, (GLsizei) strlen(code), (GLubyte *) code);

    GLint error_pos;
    glGetIntegerv(GL_PROGRAM_ERROR_POSITION_ARB, &error_pos);
    if (error_pos != -1) {
        const GLubyte *error_string;
        error_string = glGetString(GL_PROGRAM_ERROR_STRING_ARB);
        //slashspirit shrLog("Program error at position: %d\n%s\n", (int)error_pos, error_string);
        return 0;
    }
    return program_id;
}

void initOpenGL()
{
    // create pixel buffer object
    glGenBuffersARB(1, &pbo);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
    glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, width*height*sizeof(GLubyte)*4, h_img, GL_STREAM_DRAW_ARB);

    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
    // DEPRECATED: cutilSafeCall(cudaGLRegisterBufferObject(pbo));
    cutilSafeCall(cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo, 
					       cudaGraphicsMapFlagsWriteDiscard));

	glGenBuffersARB(1, &pboresult);
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pboresult);
	glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, width*height*scale*scale*sizeof(GLubyte)*4, h_result, GL_STREAM_DRAW_ARB);
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
	cutilSafeCall(cudaGraphicsGLRegisterBuffer(&cuda_pboresult_resource, pboresult,
						   cudaGraphicsMapFlagsWriteDiscard));
    
    // create texture for display
    glGenTextures(1, &texid);
    glBindTexture(GL_TEXTURE_2D, texid);
   	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindTexture(GL_TEXTURE_2D, 0);


	glGenTextures(1, &texidresult);
	glBindTexture(GL_TEXTURE_2D, texidresult);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width*scale, height*scale, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindTexture(GL_TEXTURE_2D, 0);

    // load shader program
    shader = compileASMShader(GL_FRAGMENT_PROGRAM_ARB, shader_code);
}


void initGL( int *argc, char **argv )
{
    // initialize GLUT
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(768, 768);
    glutCreateWindow("CUDA Depixelizer");
    glutDisplayFunc(display);

    glutKeyboardFunc(keyboard);
    glutReshapeFunc(reshape);
	glutTimerFunc(REFRESH_DELAY, timerEvent, 0);


    glewInit();
    //if (g_bFBODisplay) {
     //   if (!glewIsSupported( "GL_VERSION_2_0 GL_ARB_fragment_program GL_EXT_framebuffer_object" )) {
     //       shrLog("Error: failed to get minimal extensions for demo\n");
     //       shrLog("This sample requires:\n");
      //      shrLog("  OpenGL version 2.0\n");
       //     shrLog("  GL_ARB_fragment_program\n");
        //    shrLog("  GL_EXT_framebuffer_object\n");
          //  exit(-1);
        //}
    //} else {
      if (!glewIsSupported( "GL_VERSION_2_0 GL_ARB_fragment_program GL_EXT_framebuffer_object" )) {
        //    shrLog("Error: failed to get minimal extensions for demo\n");
        //    shrLog("This sample requires:\n");
          //  shrLog("  OpenGL version 1.5\n");
           // shrLog("  GL_ARB_vertex_buffer_object\n");
           // shrLog("  GL_ARB_pixel_buffer_object\n");
            exit(-1);
        }
    //}
}



void loadImageData(int argc, char **argv)
{
    // load image (needed so we can get the width and height before we create the window
    char* image_path = NULL;
    if (argc >= 1) image_path = shrFindFilePath(image_filename, argv[0]);
    if (image_path == 0) {
      //  shrLog("Error finding image file '%s'\n", image_filename);
        exit(EXIT_FAILURE);
    }
	
	cutilCheckError(cutLoadPPM4ub(image_path, (unsigned char **) &h_img, &width, &height));
    if (!h_img) {
      //  shrLog("Error opening file '%s'\n", image_path);
        exit(-1);
    }

	h_result = (unsigned int*)calloc(0, width*height*scale*scale*sizeof(unsigned int));

    //shrLog("Loaded '%s', %d x %d pixels\n\n", image_path, width, height);
}

bool checkCUDAProfile(int dev, int min_runtime, int min_compute)
{
    int runtimeVersion = 0;     

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);

    fprintf(stderr,"\nDevice %d: \"%s\"\n", dev, deviceProp.name);
    cudaRuntimeGetVersion(&runtimeVersion);
    fprintf(stderr,"  CUDA Runtime Version     :\t%d.%d\n", runtimeVersion/1000, (runtimeVersion%100)/10);
    fprintf(stderr,"  CUDA Compute Capability  :\t%d.%d\n", deviceProp.major, deviceProp.minor);

    if( runtimeVersion >= min_runtime && ((deviceProp.major<<4) + deviceProp.minor) >= min_compute ) {
        return true;
    } else {
        return false;
    }
}

int findCapableDevice(int argc, char **argv)
{
    int dev;
    int bestDev = -1;

    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
    if (error_id != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceCount returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id) );
    }

    if(deviceCount==0)
        fprintf(stderr,"There are no CUDA capabile devices.\n");
    else
        fprintf(stderr,"Found %d CUDA Capable device(s) supporting CUDA\n", deviceCount);

    for (dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        if( checkCUDAProfile( dev, MIN_RUNTIME_VERSION, MIN_COMPUTE_VERSION ) ) {
            fprintf(stderr,"\nFound CUDA Capable Device %d: \"%s\"\n", dev, deviceProp.name );
            if( bestDev == -1 ) { 
                bestDev = dev;
                fprintf(stderr, "Setting active device to %d\n", bestDev );
            }
        }
    }

    if( bestDev == -1 ) {
        fprintf(stderr, "\nNo configuration with available capabilities was found.  Test has been waived.\n");
        fprintf(stderr, "The SDK sample minimum requirements:\n");
        fprintf(stderr, "\tCUDA Compute Capability >= %d.%d is required\n", MIN_COMPUTE_VERSION/16, MIN_COMPUTE_VERSION%16);
        fprintf(stderr, "\tCUDA Runtime Version    >= %d.%d is required\n", MIN_RUNTIME_VERSION/1000, (MIN_RUNTIME_VERSION%100)/10);
 //       shrQAFinishExit(argc, (const char **)argv, QA_PASSED);
    }
    return bestDev;
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main( int argc, char** argv) 
{

    // load image to process
    loadImageData(argc, argv);


        // First initialize OpenGL context, so we can properly set the GL for CUDA.
        // This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
        initGL( &argc, argv );
	int dev = findCapableDevice(argc, argv);
	if (dev != -1) {
		cudaGLSetGLDevice( dev );
	} else {
		cutilDeviceReset();
	}
		
        initCuda();
        initOpenGL();

    atexit(cleanup);

    glutMainLoop();

    cutilDeviceReset();
}
