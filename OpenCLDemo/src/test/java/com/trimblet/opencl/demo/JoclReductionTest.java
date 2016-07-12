package com.trimblet.opencl.demo;

import java.util.function.BiFunction;

import com.trimblet.opencl.obj.OpenCLContext;

public final class JoclReductionTest extends ReductionTest {

	@Override
	public BiFunction<OpenCLContext, float[], Float> getFunction() {
		return JoclReduction::reduce;
	}

}
