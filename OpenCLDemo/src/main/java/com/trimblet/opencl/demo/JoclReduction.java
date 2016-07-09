package com.trimblet.opencl.demo;

import static org.jocl.CL.CL_CONTEXT_PLATFORM;
import static org.jocl.CL.CL_DEVICE_TYPE_ALL;
import static org.jocl.CL.CL_MEM_COPY_HOST_PTR;
import static org.jocl.CL.CL_MEM_READ_ONLY;
import static org.jocl.CL.CL_TRUE;
import static org.jocl.CL.clBuildProgram;
import static org.jocl.CL.clCreateBuffer;
import static org.jocl.CL.clCreateCommandQueue;
import static org.jocl.CL.clCreateContext;
import static org.jocl.CL.clCreateKernel;
import static org.jocl.CL.clCreateProgramWithSource;
import static org.jocl.CL.clEnqueueNDRangeKernel;
import static org.jocl.CL.clEnqueueReadBuffer;
import static org.jocl.CL.clGetDeviceIDs;
import static org.jocl.CL.clGetPlatformIDs;
import static org.jocl.CL.clReleaseCommandQueue;
import static org.jocl.CL.clReleaseContext;
import static org.jocl.CL.clReleaseKernel;
import static org.jocl.CL.clReleaseMemObject;
import static org.jocl.CL.clReleaseProgram;
import static org.jocl.CL.clSetKernelArg;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.jocl.CL;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;
import org.jocl.cl_context;
import org.jocl.cl_context_properties;
import org.jocl.cl_device_id;
import org.jocl.cl_kernel;
import org.jocl.cl_mem;
import org.jocl.cl_platform_id;
import org.jocl.cl_program;

public final class JoclReduction {

	private static final Logger LOG = LogManager.getLogger();

	private static final String PROGRAM_NAME = "reduce";
	private static final String PROGRAM_FILE = "/reduction.cl";

	//	private static final int LOCAL_WORK_SIZE = 128;
	//	private static final int NUM_WORK_GROUPS = 64;
	private static final int LOCAL_WORK_SIZE = 2;
	private static final int NUM_WORK_GROUPS = 64;

	/** The OpenCL context */
	private static cl_context context;


	/** The OpenCL command queue to which the all work will be dispatched */
	private static cl_command_queue commandQueue;


	/** The OpenCL program containing the reduction kernel */
	private static cl_program program;


	/** The OpenCL kernel that performs the reduction */
	private static cl_kernel kernel;


	/**
	 * The entry point of this sample
	 *
	 * @param args Not used
	 */
	public static void main(String args[]) {
		initialize();

		// Create input array that will be reduced
		int n = 100_000;
		//		int n = 1_000;
		//		int n = 10_000_000;
		float inputArray[] = new float[n];
		for (int i=0; i<n; i++) {
			inputArray[i] = i;
		}

		// Compute the reduction on the GPU and the CPU and print the results
		Long start = System.nanoTime();
		Float resultGPU = reduce(inputArray);
		Long end1 = System.nanoTime();
		Float resultCPU = reduceJava(inputArray);
		Long end2 = System.nanoTime();
		System.out.println(String.format("GPU: sum(%s) = %s (%sms)", n, resultGPU, ((end1 - start)/1000)));
		System.out.println(String.format("CPU: sum(%s) = %s (%sms)", n, resultCPU, ((end2 - end1)/1000)));

		shutdown();
	}


	/**
	 * Perform a reduction of the given input array on the GPU and return
	 * the result. <br />
	 * <br />
	 * The reduction is performed in two phases: In the first phase, each
	 * work group of the GPU computes the reduction of a part of the
	 * input array. The size of this part is exactly the number of work
	 * items in the group, and the reduction will be performed in local
	 * memory. The results of these reductions will be written into
	 * an output array. This output array is then reduced on the CPU.
	 *
	 * @param inputArray The array on which the reduction will be performed
	 * @return The result of the reduction
	 */
	private static float reduce(float inputArray[]) {

		float outputArray[] = new float[NUM_WORK_GROUPS];

		// Allocate the memory objects for the input- and output data
		cl_mem inputMem = clCreateBuffer(context,
				CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
				Sizeof.cl_float * inputArray.length, Pointer.to(inputArray), null);
		cl_mem outputMem = clCreateBuffer(context,
				CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
				Sizeof.cl_float * NUM_WORK_GROUPS, Pointer.to(outputArray), null);

		// Perform the reduction on the GPU: Each work group will
		// perform the reduction of 'localWorkSize' elements, and
		// the results will be written into the output memory
		reduce(inputMem, inputArray.length,
				outputMem, NUM_WORK_GROUPS,
				LOCAL_WORK_SIZE);

		// Read the output data
		clEnqueueReadBuffer(commandQueue, outputMem, CL_TRUE, 0,
				NUM_WORK_GROUPS * Sizeof.cl_float, Pointer.to(outputArray),
				0, null, null);

		// Perform the final reduction, by reducing the results
		// from the work groups on the CPU
		float result = reduceJava(outputArray);

		// Release memory objects
		clReleaseMemObject(inputMem);
		clReleaseMemObject(outputMem);

		return result;
	}


	/**
	 * Perform a reduction of the float elements in the given input memory.
	 * Each work group will reduce 'localWorkSize' elements, and write the
	 * result into the given output memory.
	 *
	 * @param inputMem The input memory containing the float values to reduce
	 * @param n The number of values in the input memory
	 * @param outputMem The output memory that will store the reduction
	 * result for each work group
	 * @param numWorkGroups The number of work groups
	 * @param localWorkSize The local work size, that is, the number of
	 * work items in each work group
	 */
	private static void reduce(
			cl_mem inputMem, int n,
			cl_mem outputMem, int numWorkGroups,
			int localWorkSize) {
		// Set the arguments for the kernel
		int a = 0;
		clSetKernelArg(kernel, a++, Sizeof.cl_mem, Pointer.to(inputMem));
		clSetKernelArg(kernel, a++, Sizeof.cl_float * localWorkSize, null);
		clSetKernelArg(kernel, a++, Sizeof.cl_int, Pointer.to(new int[]{n}));
		clSetKernelArg(kernel, a++, Sizeof.cl_mem, Pointer.to(outputMem));

		// Compute the number of work groups and the global work size
		final long globalWorkSize = numWorkGroups * localWorkSize;

		// Execute the kernel

		clEnqueueNDRangeKernel(commandQueue, kernel, 1, null,
				new long[]{globalWorkSize}, null, 0, null, null);
	}


	/**
	 * Implementation of a Kahan summation reduction in plain Java
	 *
	 * @param array The input
	 * @return The reduction result
	 */
	private static float reduceJava(float array[])
	{
		float sum = array[0];
		float c = 0.0f;
		for (int i = 1; i < array.length; i++) {
			float y = array[i] - c;
			float t = sum + y;
			c = (t - sum) - y;
			sum = t;
		}
		return sum;
	}


	/**
	 * Initialize a default OpenCL context, command queue, program and kernel
	 */
	private static void initialize() {
		// The platform, device type and device number
		// that will be used
		final int platformIndex = 0;
		final long deviceType = CL_DEVICE_TYPE_ALL;
		final int deviceIndex = 0;

		// Enable exceptions and subsequently omit error checks in this sample
		CL.setExceptionsEnabled(true);

		// Obtain the number of platforms
		int numPlatformsArray[] = new int[1];
		clGetPlatformIDs(0, null, numPlatformsArray);
		int numPlatforms = numPlatformsArray[0];

		// Obtain a platform ID
		cl_platform_id platforms[] = new cl_platform_id[numPlatforms];
		clGetPlatformIDs(platforms.length, platforms, null);
		cl_platform_id platform = platforms[platformIndex];

		// Initialize the context properties
		cl_context_properties contextProperties = new cl_context_properties();
		contextProperties.addProperty(CL_CONTEXT_PLATFORM, platform);

		// Obtain the number of devices for the platform
		int numDevicesArray[] = new int[1];
		clGetDeviceIDs(platform, deviceType, 0, null, numDevicesArray);
		int numDevices = numDevicesArray[0];

		// Obtain a device ID
		cl_device_id devices[] = new cl_device_id[numDevices];
		clGetDeviceIDs(platform, deviceType, numDevices, devices, null);
		cl_device_id device = devices[deviceIndex];

		// Create a context for the selected device
		context = clCreateContext(
				contextProperties, 1, new cl_device_id[]{device},
				null, null, null);

		// Create a command-queue for the selected device
		commandQueue = clCreateCommandQueue(context, device, 0, null);

		// Create the program from the source code
		String programSource = readFile(PROGRAM_FILE);
		program = clCreateProgramWithSource(context,
				1, new String[]{ programSource }, null, null);

		// Build the program
		clBuildProgram(program, 0, null, null, null, null);

		// Create the kernel
		kernel = clCreateKernel(program, PROGRAM_NAME, null);
	}


	/**
	 * Shut down and release all resources that have been allocated
	 * in {@link #initialize()}
	 */
	private static void shutdown() {
		clReleaseKernel(kernel);
		clReleaseProgram(program);
		clReleaseCommandQueue(commandQueue);
		clReleaseContext(context);
	}


	/**
	 * Read the contents of the file with the given name, and return
	 * it as a string
	 *
	 * @param path The name of the file to read
	 * @return The contents of the file
	 */
	private static String readFile(String path) {
		StringBuilder sb = new StringBuilder();
		try (InputStream stream = JoclReduction.class.getResourceAsStream(path)) {
			if (stream != null) {
				try (BufferedReader br = new BufferedReader(new InputStreamReader(stream))) {
					String line = null;
					while ((line = br.readLine()) != null) {
						sb.append(line);
						sb.append("\n");
					}
				}
			} else {
				LOG.error("Couldn't find specified program: " + path);
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
		return sb.toString();
	}

}
