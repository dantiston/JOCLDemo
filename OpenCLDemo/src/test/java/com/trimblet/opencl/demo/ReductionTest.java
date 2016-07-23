package com.trimblet.opencl.demo;

import static org.junit.Assert.assertEquals;

import java.util.function.BiFunction;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.junit.Test;

import com.trimblet.opencl.constants.Constants;
import com.trimblet.opencl.obj.OpenCLContext;
import com.trimblet.opencl.utilities.Utilities;

public abstract class ReductionTest {

	private static final Float CONFIDENCE_INTERVAL = 0.0001f;

	public abstract BiFunction<OpenCLContext, float[], Float> getFunction();

	@Test
	public final void testReduction() {
		try (OpenCLContext context = new OpenCLContext(Constants.PROGRAM_FILE, Constants.PROGRAM_NAME)) {
			float result = this.getFunction().apply(context, Utilities.newTestArray(5));
			assertEquals(15.0f, result, CONFIDENCE_INTERVAL);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
}
