package com.trimblet.opencl.obj.arrays;

import org.jocl.Pointer;
import org.jocl.cl_context;

/**
 * Implementation of {@link OpenCLArray} storing {@link int} objects
 *
 * @author trimblet
 */
public final class OpenCLIntArray extends OpenCLArray {

	public OpenCLIntArray(int[] array, int bytesRequired, cl_context openCLContext) {
		super(Pointer.to(array), array.length, bytesRequired, openCLContext);
	}
}
