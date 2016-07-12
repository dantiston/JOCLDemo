package com.trimblet.opencl.obj.arrays;

import org.jocl.Pointer;
import org.jocl.cl_context;

/**
 * Implementation of {@link OpenCLArray} storing {@link float} objects
 *
 * @author trimblet
 */
public final class OpenCLFloatArray extends OpenCLArray {

	public OpenCLFloatArray(float[] array, int bytesRequired, cl_context openCLContext) {
		super(Pointer.to(array), array.length, bytesRequired, openCLContext);
	}
}
