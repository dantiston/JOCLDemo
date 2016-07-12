package com.trimblet.opencl.demo;

import static org.jocl.CL.CL_TRUE;
import static org.jocl.CL.clEnqueueNDRangeKernel;
import static org.jocl.CL.clEnqueueReadBuffer;
import static org.jocl.CL.clSetKernelArg;

import java.io.IOException;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.jocl.Pointer;
import org.jocl.Sizeof;

import com.trimblet.opencl.constants.Constants;
import com.trimblet.opencl.obj.OpenCLContext;
import com.trimblet.opencl.obj.arrays.OpenCLArray;
import com.trimblet.opencl.obj.arrays.OpenCLFloatArray;
import com.trimblet.opencl.utilities.Utilities;

public final class JoclReduction {

	private static final Logger LOG = LogManager.getLogger();

	// Maxing out at 16 on mba
	// TODO: Changing this makes the answer too small...
	// Because this is being automatically generated... maybe this needs to be 1?
	private static final int LOCAL_WORK_SIZE = 1;
	private static final int NUM_WORK_GROUPS = 64;
	private static final int WORK_DIMENSIONS = 1;


	/**
	 * The entry point of this sample
	 *
	 * @param args Not used
	 * @throws IOException
	 */
	public static void main(String args[]) throws Exception {

		try (OpenCLContext context = new OpenCLContext(Constants.PROGRAM_FILE, Constants.PROGRAM_NAME)) {

			// Create input array that will be reduced
			int n = 1_000;
			float[] inputArray = Utilities.newTestArray(n);

			// Compute the reduction on the GPU and the CPU and print the results
			Long start = System.nanoTime();
			Float resultGPU = reduce(context, inputArray);
			Long end1 = System.nanoTime();
			Float resultCPU = reduceJava(inputArray);
			Long end2 = System.nanoTime();
			System.out.println(String.format("GPU: sum(%s) = %s (%sms)", n, resultGPU, ((end1 - start)/1000)));
			System.out.println(String.format("CPU: sum(%s) = %s (%sms)", n, resultCPU, ((end2 - end1)/1000)));
		}
	}


	/**
	 * Perform a reduction of the given input array on the GPU and return
	 * the result.<br/>
	 * <br/>
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
	public static float reduce(OpenCLContext context, float[] inputArray) {

		float result = -1.0f;
		float[] outputArray = new float[NUM_WORK_GROUPS];


		// Allocate the memory objects for the input- and output data
		try (OpenCLArray inputMem = new OpenCLFloatArray(inputArray, Sizeof.cl_float * inputArray.length, context.getContext());
				OpenCLArray outputMem = new OpenCLFloatArray(outputArray, Sizeof.cl_float * NUM_WORK_GROUPS, context.getContext())) {

			// Perform the reduction on the GPU: Each work group will
			// perform the reduction of 'localWorkSize' elements, and
			// the results will be written into the output memory
			reduceIntoArray(context, inputMem, outputMem, NUM_WORK_GROUPS, LOCAL_WORK_SIZE);

			// Read the output data
			clEnqueueReadBuffer(context.getQueue(), outputMem.get(), CL_TRUE, 0,
					NUM_WORK_GROUPS * Sizeof.cl_float, outputMem.getPointer(),
					0, null, null);

			// Perform the final reduction, by reducing the results
			// from the work groups on the CPU
			result = reduceJava(outputArray);
		} catch (IOException e) {
			e.printStackTrace();
		}

		return result;
	}


	/**
	 * Perform a reduction of the float elements in the given input memory.
	 * Each work group will reduce 'localWorkSize' elements, and write the
	 * result into the given output memory.
	 *
	 * @param context the {@link OpenCLContext} object wrapping this action
	 * @param inputMem The input memory containing the float values to reduce
	 * @param outputMem The output memory that will store the reduction
	 * result for each work group
	 * @param numWorkGroups The number of work groups
	 * @param localWorkSize The local work size, that is, the number of
	 * work items in each work group
	 */
	private static void reduceIntoArray(
			OpenCLContext context,
			OpenCLArray inputMem,
			OpenCLArray outputMem,
			int numWorkGroups,
			int localWorkSize) {
		// Set the arguments for the kernel
		int a = 0;
		// Must make new Pointers here...?
		clSetKernelArg(context.getKernel(), a++, Sizeof.cl_mem, Pointer.to(inputMem.get()));
		clSetKernelArg(context.getKernel(), a++, Sizeof.cl_float * localWorkSize, null);
		clSetKernelArg(context.getKernel(), a++, Sizeof.cl_int, Pointer.to(new int[]{inputMem.size()}));
		clSetKernelArg(context.getKernel(), a++, Sizeof.cl_mem, Pointer.to(outputMem.get()));

		// Compute the number of work groups and the global work size
		long globalWorkSize = numWorkGroups * localWorkSize;

		// Execute the kernel
		clEnqueueNDRangeKernel(context.getQueue(), context.getKernel(), WORK_DIMENSIONS, null,
				new long[]{globalWorkSize}, // Global workspace
				null, // Local workspace: null tells OpenCL to use appropriate size

				// Not using event infrastructure (for now),
				// so these should be 0, null, null
				0, null, null);
	}


	/**
	 * Implementation of a Kahan summation reduction in plain Java
	 *
	 * @param array The input
	 * @return The reduction result
	 */
	private static float reduceJava(float[] array) {
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

}
