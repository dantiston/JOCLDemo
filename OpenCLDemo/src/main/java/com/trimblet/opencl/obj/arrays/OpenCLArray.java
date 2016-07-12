package com.trimblet.opencl.obj.arrays;

import static org.jocl.CL.CL_MEM_COPY_HOST_PTR;
import static org.jocl.CL.CL_MEM_READ_ONLY;
import static org.jocl.CL.clCreateBuffer;
import static org.jocl.CL.clReleaseMemObject;

import java.io.Closeable;
import java.io.IOException;

import org.jocl.Pointer;
import org.jocl.cl_context;
import org.jocl.cl_mem;


/**
 * Basic wrapper implementation around {@link cl_mem}
 * implementing the {@link Closeable} interface to provide
 * auto-release of memory. See subtypes for specifics.
 *
 * @author trimblet
 */
public abstract class OpenCLArray implements AutoCloseable {

	private final cl_mem items;
	private final int size;
	private final Pointer arrayPointer;

	protected OpenCLArray(Pointer arrayPointer, int size, int bytesRequired, cl_context openCLContext) {
		if (arrayPointer == null || openCLContext == null) {
			throw new NullPointerException("OpenCLArray#() passed null parameter");
		}
		this.arrayPointer = arrayPointer;
		this.size = size;
		this.items = clCreateBuffer(openCLContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytesRequired, arrayPointer, null);
	}

	public final cl_mem get() {
		return this.items;
	}

	public final Pointer getPointer() {
		return this.arrayPointer;
	}

	@Override
	public void close() throws IOException {
		clReleaseMemObject(this.items);
	}

	public final int size() {
		return this.size;
	}

}
