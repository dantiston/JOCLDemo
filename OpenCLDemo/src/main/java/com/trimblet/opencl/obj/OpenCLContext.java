package com.trimblet.opencl.obj;

import static org.jocl.CL.CL_CONTEXT_PLATFORM;
import static org.jocl.CL.CL_DEVICE_TYPE_ALL;
import static org.jocl.CL.clBuildProgram;
import static org.jocl.CL.clCreateCommandQueue;
import static org.jocl.CL.clCreateContext;
import static org.jocl.CL.clCreateKernel;
import static org.jocl.CL.clCreateProgramWithSource;
import static org.jocl.CL.clGetDeviceIDs;
import static org.jocl.CL.clGetPlatformIDs;
import static org.jocl.CL.clReleaseCommandQueue;
import static org.jocl.CL.clReleaseContext;
import static org.jocl.CL.clReleaseKernel;
import static org.jocl.CL.clReleaseProgram;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.jocl.CL;
import org.jocl.cl_command_queue;
import org.jocl.cl_context;
import org.jocl.cl_context_properties;
import org.jocl.cl_device_id;
import org.jocl.cl_kernel;
import org.jocl.cl_platform_id;
import org.jocl.cl_program;

import com.trimblet.opencl.demo.JoclReduction;

public final class OpenCLContext implements AutoCloseable {

	private static final Logger LOG = LogManager.getLogger();

	/** The OpenCL context */
	private final cl_context context;
	/** The OpenCL command queue to which the all work will be dispatched */
	private final cl_command_queue commandQueue;
	/** The OpenCL program containing the reduction kernel */
	private final cl_program program;
	/** The OpenCL kernel that performs the reduction */
	private final cl_kernel kernel;


	public OpenCLContext(String programFile, String programName) {
		if (programFile == null || programName == null) {
			throw new NullPointerException();
		}

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
		this.context = clCreateContext(
				contextProperties, 1, new cl_device_id[]{device},
				null, null, null);

		// Create a command-queue for the selected device
		this.commandQueue = clCreateCommandQueue(context, device, 0, null);

		// Create the program from the source code
		String programSource = readFile(programFile);
		this.program = clCreateProgramWithSource(this.context,
				1, new String[]{ programSource }, null, null);

		// Build the program
		clBuildProgram(this.program, 0, null, null, null, null);

		// Create the kernel
		this.kernel = clCreateKernel(this.program, programName, null);
	}


	public final cl_context getContext() {
		return this.context;
	}

	public final cl_kernel getKernel() {
		return this.kernel;
	}

	public final cl_program getProgram() {
		return this.program;
	}

	public final cl_command_queue getQueue() {
		return this.commandQueue;
	}


	@Override
	public void close() throws IOException {
		clReleaseKernel(this.kernel);
		clReleaseProgram(this.program);
		clReleaseCommandQueue(this.commandQueue);
		clReleaseContext(this.context);
	}


	/**
	 * Read the contents of the file with the given name, and return
	 * it as a string
	 *
	 * @param path The name of the file to read
	 * @return The contents of the file
	 */
	private static final String readFile(String path) {
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
